"""
Single-AI chatbot (same flow as Response.py) with SFP usage controlled by the
weighted-random method from Response_All_AI_weight_random.py:

  1. Roll a gender category (masculine/neutral/feminine) per turn from SFP_CATEGORY_WEIGHTS.
  2. Roll an intensity (moderately/strongly) per turn from SFP_INTENSITY_WEIGHTS_BY_CATEGORY.
  3. Offer that tier's particles in the prompt, with a fallback chain down to neutral.
  4. Classify the reply's final SFP (fugashi + Gemini) and regenerate up to
     SFP_ENFORCE_MAX_ATTEMPTS times if the tier doesn't match; keep the best attempt.
"""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import fugashi
from google import genai

KEEP_RECENT_TURNS = 2
SUMMARIZE_EVERY_N_TURNS = 5
ACTIVE_CHARACTER_ID = "hanako"
OPPONENT_PROFILE = {"gender_age": "20s male"}
MODEL = "gemini-3.5-flash-lite"

DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIR, "..", "data")
load_j = lambda f: json.load(open(os.path.join(DATA_DIR, f), "r", encoding="utf-8"))

TARGET_PROFILE = next(c for c in load_j("Character_Settings.json") if c["id"] == ACTIVE_CHARACTER_ID)
TAXONOMIES = {
    "sfp": load_j("SFP_Bucket_by_gender_detailed_kaExcluded.json"),
    "pronoun": load_j("Pronoun_Bucket.json"),
    "koshou": load_j("Koshou_Bucket.json"),
}

LOCKABLE = {
    "pronoun": {"feminine": ["私", "うち"], "masculine": ["俺", "僕", "私"], "neutral": ["私"]},
    "koshou": {k: ["あなた", "君"] for k in ["feminine", "masculine", "neutral"]},
}

SFP_TIER_LIST = [
    "strongly_feminine", "moderately_feminine", "neutral",
    "moderately_masculine", "strongly_masculine",
]
SFP_TIERS_BY_CATEGORY = {
    "feminine": ["strongly_feminine", "moderately_feminine"],
    "neutral": ["neutral"],
    "masculine": ["moderately_masculine", "strongly_masculine"],
}
SFP_CATEGORY_BY_TIER = {
    tier: category for category, tiers in SFP_TIERS_BY_CATEGORY.items() for tier in tiers
}

# For a speaker of this gender category, probability of rolling masculine /
# neutral / feminine SFPs for a given turn. Each dict must sum to 1.0.
SFP_CATEGORY_WEIGHTS = {
    "masculine": {"masculine": 0.35, "neutral": 0.50, "feminine": 0.15},
    "feminine": {"masculine": 0.15, "neutral": 0.50, "feminine": 0.35},
    "neutral": {"masculine": 0.0, "neutral": 1.0, "feminine": 0.0},
}

# Second-layer roll: moderately_* vs strongly_* once a turn rolls feminine/masculine.
SFP_INTENSITY_WEIGHTS_BY_CATEGORY = {
    "feminine": {"moderately": 0.85, "strongly": 0.15},
    "masculine": {"moderately": 0.92, "strongly": 0.08},
}

# Toggles for the named anti-attractor bans in build_persona_instruction.
ENABLE_NANDA_BAN = True
ENABLE_WA_BAN = True  # re-enabled: with it off, strongly_feminine (わ/のよね/かしら family)
                      # dominated feminine-category usage, mostly drifting from moderately_
                      # feminine/neutral rolls rather than being legitimately rolled

SFP_ENFORCE_MAX_ATTEMPTS = 3

# One is picked at random each turn and put in the prompt.
PIVOT_STRATEGIES = [
    "Ask a follow-up question about a specific detail in what the other person just said.",
    "Share a related but different personal experience of your own, without contrasting it against theirs.",
    "Express a mild difference of opinion or preference, but phrase it in a fresh way rather than a simple 'X is fine, but I prefer Y' contrast.",
    "Agree with or build on a specific detail they mentioned, then add a new detail of your own.",
    "Pivot naturally to a loosely related new topic.",
    #"Just react briefly to what they said with a short comment or reaction; no need to pivot or add new information this turn.",
]
# Short labels for debug output, index-matched to PIVOT_STRATEGIES.
PIVOT_STRATEGY_LABELS = [
    "follow_up_question",
    "share_experience",
    "mild_disagreement",
    "agree_and_add",
    "new_topic",
    "brief_reaction",
]

client = None
_tagger = None


def get_client():
    global client
    if not client:
        if "GENAI_API_KEY" not in os.environ:
            raise RuntimeError("GENAI_API_KEY is not set")
        client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
    return client


def get_tagger():
    global _tagger
    if _tagger is None:
        _tagger = fugashi.Tagger()
    return _tagger


def get_category(profile):
    """Return feminine, masculine, or neutral for this profile."""
    if (explicit := profile.get("gender_category")) in ("feminine", "masculine", "neutral"):
        return explicit
    text = " ".join(str(v) for v in profile.values()).lower()
    if any(w in text for w in ("female", "woman", "女性", "女")): return "feminine"
    if any(w in text for w in ("male", "man", "男性", "男")): return "masculine"
    return "neutral"


def find_lockable_form(candidates, reply):
    """First candidate appearing in reply as its own token, not fused into a larger word (e.g. not 私物 for 私)."""
    surfaces = {word.surface for word in get_tagger()(reply)}
    return next((w for w in candidates if w in surfaces), None)


# ==============================================================
# SFP weighted-random control (copied from Response_All_AI_weight_random.py)
# ==============================================================

def roll_sfp_intensity(rolled_category):
    weights = SFP_INTENSITY_WEIGHTS_BY_CATEGORY.get(rolled_category, SFP_INTENSITY_WEIGHTS_BY_CATEGORY["masculine"])
    intensities = list(weights.keys())
    return random.choices(intensities, weights=[weights[i] for i in intensities], k=1)[0]


def sfp_fallback_chain(rolled_category, rolled_intensity="moderately"):
    """'V'-shaped chain from the rolled tier down to neutral; never crosses to the opposite gender's tiers."""
    other_intensity = "strongly" if rolled_intensity == "moderately" else "moderately"
    if rolled_category == "feminine":
        return [f"{rolled_intensity}_feminine", f"{other_intensity}_feminine", "neutral"]
    if rolled_category == "masculine":
        return [f"{rolled_intensity}_masculine", f"{other_intensity}_masculine", "neutral"]
    return ["neutral"]


def build_sfp_instruction(t_cat):
    """Roll this turn's SFP tier and build the SFP prompt block.
    Returns (sfp_inst, rolled_cat, primary_tier)."""
    weights = SFP_CATEGORY_WEIGHTS.get(t_cat, SFP_CATEGORY_WEIGHTS["neutral"])
    categories = list(weights.keys())
    rolled_cat = random.choices(categories, weights=[weights[c] for c in categories], k=1)[0]
    rolled_intensity = roll_sfp_intensity(rolled_cat)

    # Cap intensity at moderately when the roll crosses to the speaker's OPPOSITE gender category.
    is_opposite_gender_roll = (
        t_cat in ("feminine", "masculine")
        and rolled_cat in ("feminine", "masculine")
        and rolled_cat != t_cat
    )
    if is_opposite_gender_roll and rolled_intensity == "strongly":
        rolled_intensity = "moderately"

    fallback_tiers = sfp_fallback_chain(rolled_cat, rolled_intensity)
    primary_tier = fallback_tiers[0]
    sfp_pool = TAXONOMIES["sfp"].get(primary_tier, [])
    if not sfp_pool:
        return "", rolled_cat, primary_tier

    sfps = "、".join(sfp_pool)
    fallback_steps = [
        f"step {i}: ({'、'.join(TAXONOMIES['sfp'][tier])})"
        for i, tier in enumerate(fallback_tiers[1:], start=2)
        if TAXONOMIES["sfp"].get(tier)
    ]
    fallback_clause = (
        f" If so, move down this fallback order, trying the earlier steps before the later ones, and only go "
        f"as far down as you need to sound natural: {' then '.join(fallback_steps)}."
        if fallback_steps else ""
    )

    # The two intensities within one gender side share near-identical surface patterns,
    # so name the sibling tier explicitly or the model treats them as interchangeable.
    intensity_warning = ""
    if rolled_cat != "neutral" and len(fallback_tiers) > 1:
        sibling_tier = fallback_tiers[1]
        sibling_pool = "、".join(TAXONOMIES["sfp"].get(sibling_tier, []))
        primary_label = primary_tier.replace("_", " ")
        sibling_label = sibling_tier.replace("_", " ")
        relative_strength = "weaker/more subdued" if primary_tier.startswith("strongly_") else "stronger/more marked"
        intensity_warning = (
            f"\nThis turn is specifically a '{primary_label}' intensity, NOT '{sibling_label}' — those are two "
            f"different strengths of the same gender register, not interchangeable. The '{sibling_label}' "
            f"forms ({sibling_pool}) can sound similar but are measurably {relative_strength}; do not reach for "
            f"them just because they sound close to something in the preferred list above. Only use a "
            f"'{sibling_label}' form if it is the specific fallback step named below."
        )

    # んだ/だよ-family is a strong default habit for this model; name and ban it outright.
    nanda_control = ""
    if ENABLE_NANDA_BAN and primary_tier not in ("moderately_masculine", "strongly_masculine"):
        nanda_control = (
            "\nDo NOT default to だよ/だね/だよね/だな/だろ/だろう/んだ/んだよ/んだね/んだよね as a generic sentence "
            "ending this turn — ALL of these (not just the んだ-contracted ones) are masculine-coded "
            "(moderately_masculine) and are not part of, or a safe substitute for, the list above. Attaching "
            "plain-form だ before ね/よ/よね, with or without ん, is the exact same habit and is banned the same "
            "way. This is a common habit to fall back on for explanatory nuance, but it is off-register here; "
            "use one of the listed particles instead, or no particle at all, rather than any だ-family ending."
        )

    # わ/わね/わよ/だわ/のよね is the mirror-image attractor.
    wa_control = ""
    if ENABLE_WA_BAN and primary_tier not in ("strongly_feminine",):
        wa_control = (
            "\nDo NOT default to わ/わね/わよ/わよね/だわ/だったわ/のよ/のよね/かしら/のかしら/のかしらね (or any "
            "ちゃう-contracted form of these, e.g. ちゃうわ/ちゃうのよね) as a generic sentence ending this turn — "
            "ALL of these are strongly feminine-coded and are not part of, or a safe substitute for, the list "
            "above. This is a common habit to fall back on as a stereotypical \"feminine\" voice, but it is "
            "off-register here; use one of the listed particles instead, or no particle at all, rather than "
            "any of these forms."
        )

    sfp_inst = (
        f"Sentence-final particle guidance: prefer particles from this list ({sfps}) when you end a sentence "
        "with a sentence-final particle—you do not need to use one on every sentence.\n"
        "STRICT RULE: use the particle exactly as listed, character for character. Do NOT insert だ/です (or "
        "any other copula) in front of it unless that copula is already written into the listed entry itself. "
        "Adding だ/です changes a particle's register even though the particle itself looks unchanged — these "
        "are NOT interchangeable: よね (plain) vs だよね (copula added) are different registers; ね vs だね are "
        "different registers; んだよね and んだね already contain だ as part of the listed entry itself, so "
        "they are a different, more casual/masculine-leaning form than plain よね/ね and must NOT be used as a "
        f"substitute when よね/ね (without だ/んだ) is what's listed above.{intensity_warning}{nanda_control}{wa_control}\n"
        f"Only reach for a different ending if every option in that list above would sound distinctly "
        f"unnatural for this specific sentence.{fallback_clause} If nothing in this entire fallback order "
        "would sound natural either, simply end the sentence without any sentence-final particle this turn, "
        "rather than forcing one in."
    )
    return sfp_inst, rolled_cat, primary_tier


# ==============================================================
# SFP classification (fugashi final-particle detection + Gemini against the taxonomy)
# ==============================================================

POS_SENTENCE_FINAL = '助詞'
POS2_SENTENCE_FINAL = '終助詞'
AUXILIARY_SFP_ENDINGS = ["んだ", "だろう", "だろ"]
COMPOUND_SFP_FALLBACK_ENDINGS = ["わな"]
PARTICLE_RUN_POS1 = ('助詞', '助動詞')
TRAILING_PUNCTUATION = "。！？!?　 "


def find_final_sfp(text: str) -> Optional[Dict[str, str]]:
    """Return the trailing run of particle/auxiliary tokens if it is a genuine sentence-final form."""
    if not isinstance(text, str) or not text.strip():
        return None
    tokens = [w for w in get_tagger()(text) if w.feature.pos1 != '補助記号']
    if not tokens:
        return None

    run = []
    for word in reversed(tokens):
        if word.feature.pos1 in PARTICLE_RUN_POS1:
            run.insert(0, word)
        else:
            break

    if run:
        surface = ''.join(w.surface for w in run)
        is_true_sfp = (run[-1].feature.pos1 == POS_SENTENCE_FINAL
                       and run[-1].feature.pos2 == POS2_SENTENCE_FINAL)
        is_known_auxiliary_ending = any(surface.endswith(e) for e in AUXILIARY_SFP_ENDINGS)
        if is_true_sfp or is_known_auxiliary_ending:
            return {'surface': surface}

    stripped = text.rstrip(TRAILING_PUNCTUATION)
    for ending in COMPOUND_SFP_FALLBACK_ENDINGS:
        if stripped.endswith(ending):
            return {'surface': ending}
    return None


def mark_sfp_in_text(text: str, surface: str) -> str:
    stripped = text.rstrip(TRAILING_PUNCTUATION)
    trailing_punct = text[len(stripped):]
    if stripped.endswith(surface):
        return stripped[: -len(surface)] + f"【{surface}】" + trailing_punct
    return text + f"【{surface}】"


def build_classification_prompt(taxonomy, utterance_text, marked_particle, preceding_context):
    taxonomy_lines = []
    for bucket, entries in taxonomy.items():
        taxonomy_lines.append(f"[{bucket}]")
        for entry in entries:
            taxonomy_lines.append(f"  - {entry}")
    taxonomy_block = "\n".join(taxonomy_lines)
    context_block = "\n".join(preceding_context) if preceding_context else "(no preceding context)"
    bucket_names = ', '.join(taxonomy.keys())

    return (
        "You are classifying a Japanese sentence-final particle (終助詞) against a "
        "gendered SFP taxonomy. Some taxonomy entries are grammatical patterns "
        "(e.g. 名詞＋よ, 形容動詞＋の, 動詞の命令形＋よ, て形＋ね) that depend on the "
        "part-of-speech or conjugation form of the word immediately before the particle, "
        "not just the particle's surface form — use the full utterance to judge this.\n\n"
        f"Taxonomy (bucket: entries):\n{taxonomy_block}\n\n"
        f"Preceding conversation context:\n{context_block}\n\n"
        f"Utterance to classify (target ending marked with 【】): {utterance_text}\n"
        f"Marked particle/ending: {marked_particle}\n\n"
        "Return ONLY a JSON object with keys: "
        f"\"bucket\" (exactly one of these bucket names: {bucket_names}, or \"none\" if nothing fits), "
        "\"matched_entry\" (the exact taxonomy entry string that best matches, or null if none fit)."
    )


def classify_sfp_with_ai(taxonomy, utterance_text, marked_particle, preceding_context) -> Dict[str, Any]:
    prompt = build_classification_prompt(taxonomy, utterance_text, marked_particle, preceding_context)
    response = get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0, "seed": 0},
    ).text.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {'bucket': None, 'matched_entry': None}


def classify_reply_sfp(reply, recent):
    """Returns (used_particle, used_tier, used_category) for one reply."""
    sfp = find_final_sfp(reply)
    if sfp is None:
        return "", "no_sfp", "no_sfp"
    preceding_context = [f"{t['role']}: {t['parts'][0]['text']}" for t in recent[-3:]]
    result = classify_sfp_with_ai(
        TAXONOMIES["sfp"], mark_sfp_in_text(reply, sfp['surface']), sfp['surface'], preceding_context)
    used_tier = result.get('bucket') or 'no_sfp'
    if used_tier not in SFP_TIER_LIST:
        used_tier = 'no_sfp'
    return sfp['surface'], used_tier, SFP_CATEGORY_BY_TIER.get(used_tier, 'no_sfp')


# だ-family endings are the runaway attractor; rank them worst among misses.
DA_FAMILY_ENDINGS = ["んだよね", "んだね", "んだよ", "んだ", "だよね", "だね", "だよ", "だな", "だろう", "だろ"]


def _is_da_family(used_particle):
    particle = used_particle or ""
    return any(particle.endswith(e) for e in DA_FAMILY_ENDINGS)


def _attempt_rank(attempt, rolled_cat):
    """Best-of-N score when no attempt matched the rolled tier exactly (higher is better):
    3 = right category, wrong intensity; 2 = neutral/no_sfp; 0 = opposite gender;
    -1 = opposite gender and だ-family."""
    if attempt["used_category"] == rolled_cat:
        return 3
    if attempt["used_category"] in ("neutral", "no_sfp"):
        return 2
    if _is_da_family(attempt.get("used_particle")):
        return -1
    return 0


# ==============================================================
# Chatbot (same flow as Response.py)
# ==============================================================

def build_persona_instruction(target, opp, opp_style):
    """Returns (system_instruction, rolled_cat, rolled_tier, pivot_strategy)."""
    t_cat, o_cat = get_category(target), get_category(opp)

    sfp_inst, rolled_cat, rolled_tier = build_sfp_instruction(t_cat)
    prons = "、".join(TAXONOMIES["pronoun"].get(t_cat, []))
    kosh = "、".join(TAXONOMIES["koshou"].get(o_cat, []))

    pron_inst = f"Self-reference pronoun guidance: prefer the {t_cat} inventory ({prons}). Choose one form and keep using the same one consistently across the conversation. Do not switch to a different self-reference pronoun after it has been established. " if prons else ""
    kosh_inst = f"Address-form guidance: prefer the {o_cat} inventory ({kosh}) for how you address the other person. Choose one address form and keep using the same one consistently across the conversation. Do not switch to a different address form after it has been established. " if kosh else ""

    personality = str(target.get("personality", "")).strip() or "neutral"
    pivot_strategy = random.choice(PIVOT_STRATEGIES)
    # Pivot guidance goes second, right after SFP: it has no retry backstop, so position is
    # its only lever. No blanket "don't end every turn with a question" rule — it contradicted
    # the follow-up-question strategy; each strategy fully specifies its own turn behavior.
    pivot_inst = (
        f"For this turn, react using this approach: {pivot_strategy} "
        "Do not overuse the pattern '[topic] wa ii kedo, watashi/boku wa [other topic]' (e.g. '〜もいいけど、私は〜') — if you used this phrasing recently, use a different approach this turn. "
    )

    return (
        f"You are {target['name']}, a {target['gender_age']} {target['job']}. "
        f"Your personality is {personality}. "
        f"Your hobby is {target['hobby']}. "
        f"Your favorite things include {target['favorite_things']}. "
        f"You are having a conversation with a {opp['gender_age']}. "
        f"The person you're talking to seems {opp_style} in how they speak — "
        "respond naturally given that. "
        "You are having a casual conversation with this person in a room."
        "Respond naturally in character, as this persona would, based on the conversation so far.\n"
        # SFP guidance comes right after the persona intro, ahead of the general style rules.
        f"{sfp_inst}\n"
        f"{pivot_inst}\n"
        f"{pron_inst}{kosh_inst}\n"
        "CRITICAL: Do not invent or assume any background details about the user's current situation, schedule, or life (e.g., do not assume they are working, studying, or on a break). "
        "At the start of the conversation, if the user only gives a short greeting, introduce yourself naturally and ask a broad, safe icebreaker—like asking for their name, or try to engage them for self introduction. "
        "Always respond in Japanese only, regardless of what language the user writes in. "
        "Keep your response short, one or two short "
        "sentences at most, not a lengthy reply. Have some follow up questions "
        "or comments to keep the conversation going, but don't ask more than one question at a time. "
        "When you do share about yourself, link it naturally to what the user just said. "
        "When you pivot to a new angle or topic, briefly acknowledge what the user just said first, rather than ignoring it and asserting an unrelated topic in parallel. "
        "Every few turns, you may circle back to something mentioned earlier in the conversation instead of only introducing new topics. "
        "Avoid simply agreeing with or complimenting what the user said (e.g. do not just say it sounds nice/wonderful/great). " #prevent yes-man responses
        "Do not repeat or re-name the specific topic word/noun the user just introduced back to them before reacting (e.g. if they say '動物のやつが好き', don't reply starting with '動物モチーフだと〜'; if they mention '足湯', don't reply starting with '足湯は〜') — just react to the substance directly without restating the keyword." #prevent parrotting
    ), rolled_cat, rolled_tier, pivot_strategy


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
    res = get_client().models.generate_content(model=MODEL, contents=prompt).text
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


def generate_response(state, user_input, debug_sfp=False):
    state["recent"].append({"role": "user", "parts": [{"text": user_input}]})

    sys_inst, rolled_cat, rolled_tier, pivot_strategy = build_persona_instruction(TARGET_PROFILE, OPPONENT_PROFILE, state["opponent_style"])
    pivot_label = PIVOT_STRATEGY_LABELS[PIVOT_STRATEGIES.index(pivot_strategy)]
    if p := state.get("pronoun_choice"):
        sys_inst += f" Established self-reference pronoun: {p}. Use this exact pronoun consistently and do not switch forms."
    if k := state.get("koshou_choice"):
        sys_inst += f" Established address form for the other person: {k}. Use this exact form consistently and do not switch forms."
    if s := state.get("summary"):
        sys_inst += f" Summary of earlier conversation so far: {s}"

    # Regenerate until the reply's final SFP lands on the rolled tier (no_sfp always
    # counts as a match — a plain ending is never wrong-gendered); otherwise keep the best attempt.
    attempts = []
    overall_start = time.perf_counter()
    for attempt_num in range(1, SFP_ENFORCE_MAX_ATTEMPTS + 1):
        attempt_start = time.perf_counter()
        reply = get_client().models.generate_content(
            model=MODEL,
            contents=state["recent"],
            config={"system_instruction": sys_inst},
        ).text.strip().replace("\n", " ")
        used_particle, used_tier, used_category = classify_reply_sfp(reply, state["recent"][:-1])
        attempt_elapsed = time.perf_counter() - attempt_start
        attempts.append({
            "reply": reply, "used_particle": used_particle,
            "used_tier": used_tier, "used_category": used_category,
            "elapsed": attempt_elapsed,
        })
        if used_tier == rolled_tier or used_category == "no_sfp":
            break
    overall_elapsed = time.perf_counter() - overall_start

    final = max(attempts, key=lambda a: _attempt_rank(a, rolled_cat))
    reply = final["reply"]
    if debug_sfp:
        attempt_times = ";".join(f"{a['elapsed']:.3f}" for a in attempts)
        print(f"[SFP DEBUG] rolled={rolled_cat} rolled_tier={rolled_tier} pivot={pivot_label} "
              f"used_particle={final['used_particle'] or '(none)'} used_tier={final['used_tier']} "
              f"attempts={len(attempts)} overall_s={overall_elapsed:.3f} attempt_s={attempt_times}")

    t_cat, o_cat = get_category(TARGET_PROFILE), get_category(OPPONENT_PROFILE)
    if not state.get("pronoun_choice"):
        state["pronoun_choice"] = find_lockable_form(LOCKABLE["pronoun"].get(t_cat, []), reply)
    if not state.get("koshou_choice"):
        state["koshou_choice"] = find_lockable_form(LOCKABLE["koshou"].get(o_cat, []), reply)

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
    import sys
    debug_sfp = "--debug" in sys.argv
    state = {"summary": "", "recent": [], "opponent_style": "neutral", "pronoun_choice": None, "koshou_choice": None}
    print("Chat with the persona. Type 'exit' or 'quit' to stop." + (" (SFP debug on)" if debug_sfp else ""))
    while (user_input := input("You: ").strip()) and user_input.lower() not in ("exit", "quit"):
        print(f"Persona: {generate_response(state, user_input, debug_sfp=debug_sfp)}")


if __name__ == '__main__':
    main()
