import json
import os
from google import genai

KEEP_RECENT_TURNS = 2
SUMMARIZE_EVERY_N_TURNS = 5
ACTIVE_CHARACTER_ID = "taro"
OPPONENT_PROFILE = {"gender_age": "40s male"}

DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIR, "..", "data")
load_j = lambda f: json.load(open(os.path.join(DATA_DIR, f), "r", encoding="utf-8"))

TARGET_PROFILE = next(c for c in load_j("Character_Settings.json") if c["id"] == ACTIVE_CHARACTER_ID)
TAXONOMIES = {
    "sfp": load_j("SFP_Bucket_by_gender.json"),
    "pronoun": load_j("Pronoun_Bucket.json"),
    "koshou": load_j("Koshou_Bucket.json")
}

LOCKABLE = {
    "pronoun": {"feminine": ["私", "うち"], "masculine": ["俺", "僕", "私"], "neutral": ["私"]},
    "koshou": {k: ["あなた", "君"] for k in ["feminine", "masculine", "neutral"]}
}

client = None

def get_client():
    global client
    if not client:
        if "GENAI_API_KEY" not in os.environ:
            raise RuntimeError("GENAI_API_KEY is not set")
        client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
    return client

def get_category(profile):
    """Infer feminine, masculine, or neutral from profile values."""
    text = " ".join(str(v) for v in profile.values()).lower()
    if any(w in text for w in ("female", "woman", "女性", "女")): return "feminine"
    if any(w in text for w in ("male", "man", "男性", "男")): return "masculine"
    return "neutral"

def build_persona_instruction(target, opp, opp_style):
    t_cat, o_cat = get_category(target), get_category(opp)
    
    sfps = "、".join(TAXONOMIES["sfp"].get(t_cat, []))
    prons = "、".join(TAXONOMIES["pronoun"].get(t_cat, []))
    kosh = "、".join(TAXONOMIES["koshou"].get(o_cat, []))
    
    sfp_inst = f"Sentence-final particle guidance: prefer the {t_cat} inventory ({sfps}). Keep the reply consistent with that inventory and avoid particles from the other categories. Use at most one sentence-final particle per sentence. " if sfps else ""
    pron_inst = f"Self-reference pronoun guidance: prefer the {t_cat} inventory ({prons}). Choose one form and keep using the same one consistently across the conversation. Do not switch to a different self-reference pronoun after it has been established. " if prons else ""
    kosh_inst = f"Address-form guidance: prefer the {o_cat} inventory ({kosh}) for how you address the other person. Choose one address form and keep using the same one consistently across the conversation. Do not switch to a different address form after it has been established. " if kosh else ""

    personality = str(target.get("personality", "")).strip() or "neutral"

    return (
        f"You are {target['name']}, a {target['gender_age']} {target['job']}. "
        f"Your personality is {personality}. "
        f"Your hobby is {target['hobby']}. "
        f"Your favorite things include {target['favorite_things']}. "
        f"You are having a conversation with a {opp['gender_age']}. "
        f"The person you're talking to seems {opp_style} in how they speak — "
        "respond naturally given that. "
        "You are having a casual conversation with this person in a room."
        "Respond naturally in character, as this persona would, based on the conversation so far. "
        "CRITICAL: Do not invent or assume any background details about the user's current situation, schedule, or life (e.g., do not assume they are working, studying, or on a break). "
        "At the start of the conversation, if the user only gives a short greeting, introduce yourself naturally and ask a broad, safe icebreaker—like asking for their name, or try to engage them for self introduction. "
        "Always respond in Japanese only, regardless of what language the user writes in. "
        "Keep your response short, one or two short "
        "sentences at most, not a lengthy reply. Have some follow up questions "
        "or comments to keep the conversation going, but don't ask more than one question at a time. "
        f"{pron_inst}{kosh_inst}{sfp_inst}"
        "Do not end every turn with a question. "
        "If you asked a question in your previous turn, just react to the user's answer and share something about your own hobbies or feelings this turn. "
        "Only ask a question if the user asks you one, or if the conversation completely stalls. "
        "When you do share about yourself, link it naturally to what the user just said."
    )

def summarize_history(summary, style, pronoun_choice, koshou_choice, aged_turns):
    text = "\n".join(f"{t['role']}: {t['parts'][0]['text']}" for t in aged_turns)
    prompt = (
        "Summarize the following conversation so far in a few concise sentences, "
        "preserving important facts and context for continuing the conversation. "
        "Also guess the opponent's (the 'user' role's) speaking style in a few words "
        "(e.g. playful, polite, friendly, blunt, formal). Also identify the self-reference pronoun "
        "the persona is using and the address form the persona is using for the opponent, if they appear in the conversation.\n\n"
        f"Previous summary:\n{summary or '(none)'}\n\n"
        f"Previous opponent style guess:\n{style}\n\n"
        f"Previous self-reference pronoun guess:\n{pronoun_choice or '(none)'}\n\n"
        f"Previous address form guess:\n{koshou_choice or '(none)'}\n\n"
        f"New turns to fold in:\n{text}\n\n"
        'Respond with only a JSON object in this exact form: '
        '{"summary": "...", "opponent_style": "...", "pronoun_choice": "...", "koshou_choice": "..."}'
    )
    res = get_client().models.generate_content(model="gemini-3.5-flash-lite", contents=prompt).text
    try:
        parsed = json.loads(res.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        return (
            parsed.get("summary", summary),
            parsed.get("opponent_style", style),
            parsed.get("pronoun_choice", pronoun_choice),
            parsed.get("koshou_choice", koshou_choice),
        )
    except Exception:
        return summary, style, pronoun_choice, koshou_choice

def generate_response(state, user_input):
    state["recent"].append({"role": "user", "parts": [{"text": user_input}]})
    
    sys_inst = build_persona_instruction(TARGET_PROFILE, OPPONENT_PROFILE, state["opponent_style"])
    if p := state.get("pronoun_choice"): 
        sys_inst += f" Established self-reference pronoun: {p}. Use this exact pronoun consistently and do not switch forms."
    if k := state.get("koshou_choice"): 
        sys_inst += f" Established address form for the other person: {k}. Use this exact form consistently and do not switch forms."
    if s := state.get("summary"): 
        sys_inst += f" Summary of earlier conversation so far: {s}"

    reply = get_client().models.generate_content(
        model="gemini-3.5-flash-lite", 
        contents=state["recent"],
        config={"system_instruction": sys_inst}
    ).text.strip()

    t_cat, o_cat = get_category(TARGET_PROFILE), get_category(OPPONENT_PROFILE)
    if not state.get("pronoun_choice"):
        state["pronoun_choice"] = next((w for w in LOCKABLE["pronoun"].get(t_cat, []) if w in reply), None)
    if not state.get("koshou_choice"):
        state["koshou_choice"] = next((w for w in LOCKABLE["koshou"].get(o_cat, []) if w in reply), None)

    state["recent"].append({"role": "model", "parts": [{"text": reply}]})

    turn_count = len(state["recent"]) // 2
    if turn_count > SUMMARIZE_EVERY_N_TURNS:
        aged_count = (turn_count - KEEP_RECENT_TURNS) * 2
        state["summary"], state["opponent_style"], state["pronoun_choice"], state["koshou_choice"] = summarize_history(
            state["summary"],
            state["opponent_style"],
            state.get("pronoun_choice"),
            state.get("koshou_choice"),
            state["recent"][:aged_count],
        )
        print("[history summarized]")
        print(state["summary"])
        state["recent"] = state["recent"][aged_count:]

    return reply

def main():
    state = {"summary": "", "recent": [], "opponent_style": "neutral", "pronoun_choice": None, "koshou_choice": None}
    print("Chat with the persona. Type 'exit' or 'quit' to stop.")
    while (user_input := input("You: ").strip()) and user_input.lower() not in ("exit", "quit"):
        print(f"Persona: {generate_response(state, user_input)}")

if __name__ == '__main__':
    main()