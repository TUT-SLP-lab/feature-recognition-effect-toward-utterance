import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import fugashi
from google import genai

DEFAULT_TURNS = 30
DEFAULT_SPEAKER_IDS = ("taro", "hanako")

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


def load_j(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as handle:
        return json.load(handle)


CHARACTERS = load_j("Character_Settings.json")
TAXONOMIES = {
    "sfp": load_j("SFP_Bucket_by_gender_kaExclude.json"),
    "pronoun": load_j("Pronoun_Bucket.json"),
    "koshou": load_j("Koshou_Bucket.json"),
}

LOCKABLE = {
    "pronoun": {"feminine": ["私", "うち"], "masculine": ["俺", "僕", "私"], "neutral": ["私"]},
    "koshou": {k: ["あなた", "君"] for k in ["feminine", "masculine", "neutral"]},
}

PIVOT_STRATEGIES = [
    "Ask a follow-up question about a specific detail in what the other person just said.",
    "Share a related but different personal experience of your own, without contrasting it against theirs.",
    "Express a mild difference of opinion or preference, but phrase it in a fresh way rather than a simple 'X is fine, but I prefer Y' contrast.",
    "Agree with or build on a specific detail they mentioned, then add a new detail of your own.",
    "Pivot naturally to a loosely related new topic.",
    "Just react briefly to what they said with a short comment or reaction; no need to pivot or add new information this turn.",
]

client = None


def get_client():
    global client
    if not client:
        if "GENAI_API_KEY" not in os.environ:
            raise RuntimeError("GENAI_API_KEY is not set")
        client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
    return client


def get_category(profile):
    """Return feminine, masculine, or neutral for this profile."""
    if (explicit := profile.get("gender_category")) in ("feminine", "masculine", "neutral"):
        return explicit
    text = " ".join(str(v) for v in profile.values()).lower()
    if any(word in text for word in ("female", "woman", "女性", "女")):
        return "feminine"
    if any(word in text for word in ("male", "man", "男性", "男")):
        return "masculine"
    return "neutral"


_tagger = fugashi.Tagger()


def find_lockable_form(candidates, reply):
    """Return the first candidate appearing in reply as its own token, not fused into a larger word (e.g. not 私物 for 私)."""
    surfaces = {word.surface for word in _tagger(reply)}
    return next((w for w in candidates if w in surfaces), None)


def get_character(character_id):
    return next(character for character in CHARACTERS if character["id"] == character_id)


def select_characters(speaker_a_id, speaker_b_id):
    if speaker_a_id and speaker_b_id:
        return get_character(speaker_a_id), get_character(speaker_b_id)
    if len(CHARACTERS) < 2:
        raise RuntimeError("Character_Settings.json needs at least two characters")
    return CHARACTERS[0], CHARACTERS[1]


def build_persona_instruction(target, opponent, pronoun_lock=None, koshou_lock=None, recent_endings=None):
    t_cat, o_cat = get_category(target), get_category(opponent)

    # 1. Dynamically sample 3 SFPs for this specific turn to force variety.
    # Pool combines the character's gender bucket with the neutral bucket
    # (deduped) so gender-marked and neutral particles can both appear.
    gender_sfps = TAXONOMIES["sfp"].get(t_cat, [])
    neutral_sfps = TAXONOMIES["sfp"].get("neutral", [])
    full_sfp_list = list(dict.fromkeys(gender_sfps + neutral_sfps))
    if full_sfp_list:
        sampled_sfps = random.sample(full_sfp_list, min(3, len(full_sfp_list)))
        sfps = "、".join(sampled_sfps)
        sfp_inst = (
            f"Sentence-final particle guidance: When using a sentence-final particle, pick from this specific subset: ({sfps}). "
            "You do not need to use a marked particle on every sentence—ending naturally on a standard verb or adjective is perfectly fine."
        )
    else:
        sfp_inst = ""

    prons = "、".join(TAXONOMIES["pronoun"].get(t_cat, []))
    kosh = "、".join(TAXONOMIES["koshou"].get(o_cat, []))

    if pronoun_lock:
        pron_inst = f"Established self-reference pronoun: {pronoun_lock}. Use this exact pronoun consistently and do not switch forms."
    else:
        pron_inst = (
            f"Self-reference pronoun guidance: prefer the {t_cat} inventory ({prons}). "
            "Choose one form and keep using the same one consistently."
            if prons else ""
        )

    if koshou_lock:
        kosh_inst = (
            f"Established address form for the other person: {koshou_lock}. Use this exact form when you do address them by name/title, "
            "but only do so occasionally (roughly every 3-4 turns) — natural conversation drops it most of the time once it's established."
        )
    else:
        kosh_inst = (
            f"Address-form guidance: prefer the {o_cat} inventory ({kosh}) for how you address the other person. "
            "Choose one address form and keep using the same one consistently, but use it only occasionally (roughly every 3-4 turns), not every turn."
            if kosh else ""
        )

    pivot_strategy = random.choice(PIVOT_STRATEGIES)

    if recent_endings:
        endings_list = "、".join(f"「{e}」" for e in recent_endings)
        repetition_inst = f"Avoid ending your reply the same way as these recent endings: {endings_list}. Vary your sentence-final phrasing this turn."
    else:
        repetition_inst = ""

    # ... rest of your prompt string ...

    personality = str(target.get("personality", "")).strip() or "neutral"

    return (
        f"You are {target['name']}, a {target['gender_age']} {target['job']}. "
        f"Your personality is {personality}. "
        f"Your hobby is {target['hobby']}. "
        f"Your favorite things include {target['favorite_things']}. "
        f"You are talking with a {opponent['gender_age']}."
        "You are having a casual conversation in a room."
        "Respond naturally in character, as this persona would, based on the conversation so far. "
        "CRITICAL: Do not invent or assume any background details about the other character's situation, schedule, or life. "

        "If the conversation is just starting, introduce yourself naturally and ask a broad, safe icebreaker. "
        "Always respond in Japanese only, regardless of what language the other character uses. "
        "Keep your response short, one or two short sentences at most. "
        "Do not end every turn with a question, but you may ask one to naturally explore what the other person just said. "
        "Avoid simply agreeing with or complimenting what the other person said (e.g. do not just say it sounds nice/wonderful/great). "
        "When you pivot to a new angle or topic, briefly acknowledge what the other person just said first, rather than ignoring it and asserting an unrelated topic in parallel. "
        f"For this turn, react using this approach: {pivot_strategy} "
        "Do not overuse the pattern '[topic] wa ii kedo, watashi/boku wa [other topic]' (e.g. '〜もいいけど、私は〜') — if you used this phrasing recently, use a different approach this turn. "
        "Every few turns, you may circle back to something mentioned earlier in the conversation instead of only introducing new topics. "
        f"{repetition_inst} "
        #"When you do share about yourself, link it naturally to what the other character just said. "
        f"{pron_inst}{kosh_inst}{sfp_inst}"
    )


def transcript_text_for_speaker(turns, current_speaker_id):
    """Formats the conversation history relative to who is currently speaking."""
    if not turns:
        return "(The conversation has just started.)"
    
    formatted_lines = []
    for turn in turns:
        if turn['speaker_id'] == current_speaker_id:
            label = "You"
        else:
            label = "Other person"
        formatted_lines.append(f"{label}: {turn['text']}")
        
    return "\n".join(formatted_lines)


def clean_reply(reply, speaker_name):
    reply = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    for prefix in (f"{speaker_name}:", f"{speaker_name}："):
        if reply.startswith(prefix):
            reply = reply[len(prefix):].strip()
    return reply.replace("\n", " ").strip()


def generate_reply(speaker, opponent, turns, pronoun_lock=None, koshou_lock=None, recent_endings=None):
    system_instruction = build_persona_instruction(speaker, opponent, pronoun_lock, koshou_lock, recent_endings)

    # Format transcript from 'speaker's point of view
    history = transcript_text_for_speaker(turns, speaker['id'])

    prompt = (
        f"Conversation history:\n{history}\n\n"
        "It is your turn to speak. Reply naturally in character. "
        "Output ONLY your spoken utterance in Japanese. Do not include your name or any prefixes."
    )

    start = time.perf_counter()
    reply = get_client().models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={"system_instruction": system_instruction},
    ).text.strip()
    elapsed = time.perf_counter() - start

    return clean_reply(reply, speaker["name"]), elapsed

def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["turn", "speaker_id", "speaker_name", "listener_id", "listener_name", "text", "generation_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


RECENT_ENDINGS_KEPT = 3
ENDING_FINGERPRINT_CHARS = 10


def run_chat(speaker_a, speaker_b, turns, output_path):
    transcript = []
    rows = []
    locks = {
        speaker_a["id"]: {"pronoun": None, "koshou": None},
        speaker_b["id"]: {"pronoun": None, "koshou": None},
    }
    recent_endings = {speaker_a["id"]: [], speaker_b["id"]: []}

    for turn_index in range(turns):
        speaker = speaker_a if turn_index % 2 == 0 else speaker_b
        opponent = speaker_b if speaker is speaker_a else speaker_a
        speaker_locks = locks[speaker["id"]]
        opponent_locks = locks[opponent["id"]]
        reply, elapsed = generate_reply(
            speaker, opponent, transcript,
            pronoun_lock=speaker_locks["pronoun"],
            koshou_lock=opponent_locks["koshou"],
            recent_endings=recent_endings[speaker["id"]],
        )

        if speaker_locks["pronoun"] is None:
            speaker_locks["pronoun"] = find_lockable_form(
                LOCKABLE["pronoun"].get(get_category(speaker), []), reply
            )
        if opponent_locks["koshou"] is None:
            opponent_locks["koshou"] = find_lockable_form(
                LOCKABLE["koshou"].get(get_category(opponent), []), reply
            )

        speaker_endings = recent_endings[speaker["id"]]
        speaker_endings.append(reply[-ENDING_FINGERPRINT_CHARS:])
        del speaker_endings[:-RECENT_ENDINGS_KEPT]

        entry = {
            "turn": turn_index + 1,
            "speaker_id": speaker["id"],
            "speaker_name": speaker["name"],
            "listener_id": opponent["id"],
            "listener_name": opponent["name"],
            "text": reply,
            "generation_seconds": round(elapsed, 3),
        }
        transcript.append(entry)
        rows.append(entry)
        print(f"{speaker['name']}: {reply}")

    write_csv(rows, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a two-character AI conversation and save it as CSV.")
    parser.add_argument("--speaker-a", default=DEFAULT_SPEAKER_IDS[0], help="First character id from Character_Settings.json")
    parser.add_argument("--speaker-b", default=DEFAULT_SPEAKER_IDS[1], help="Second character id from Character_Settings.json")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="Total number of utterances to generate")
    parser.add_argument(
        "--output",
        default=f"chat_logs/chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="CSV file to write the conversation to",
    )
    args = parser.parse_args()

    speaker_a, speaker_b = select_characters(args.speaker_a, args.speaker_b)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = OUTPUTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.turns} turns between {speaker_a['name']} and {speaker_b['name']}...")
    result = run_chat(speaker_a, speaker_b, args.turns, output_path)
    print(f"Saved chat log to {result}")


if __name__ == '__main__':
    main()