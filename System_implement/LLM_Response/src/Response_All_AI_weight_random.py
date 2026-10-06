#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM_Response: weighted-random SFP persona chat generator + counter.

One file, three function groups:
  GENERATE — drive a persona-vs-persona Gemini chat, with sentence-final
             particle (SFP) usage steered by a two-layer weighted roll
             (gender category, then strongly/moderately intensity).
  DEBUG    — classify which SFP a reply actually used. Shared by both the
             live per-turn debug output during generation and the post-hoc
             counter below, so there is exactly one classification code path
             (no risk of the two disagreeing/biasing each other).
  COUNTER  — pool saved chat_logs CSVs across runs, classify every
             utterance-final SFP with the DEBUG classifier, and chart the
             resulting gender-bucket distribution per run/speaker.

CLI: --mode generate|debug|counter. generate/debug take the existing
--age-a/--gender-a/--age-b/--gender-b (or sweep) flags; debug additionally
prints a classification line per turn and saves a sibling *_sfp_debug.csv.
Both generate and debug automatically run the counter step afterward,
comparing the just-generated run (as 'weight_random') against the locked
'give_all_sfp' baseline folder — see BASELINE_RUN_FOLDER.
"""

import argparse
import csv
import json
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fugashi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from google import genai

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif',
]

# ==============================================================
# SHARED SETUP — paths, taxonomy, profiles, client/tagger singletons
# ==============================================================
DEFAULT_TURNS = 30

DIR = Path(__file__).resolve().parent
ROOT = DIR.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
CHAT_LOGS_DIR = OUTPUTS_DIR / "chat_logs"
COUNTER_OUTPUT_DIR = OUTPUTS_DIR / "sfp_ai_output"

AI_CLASSIFY_MAX_WORKERS = 8
POS_SENTENCE_FINAL = '助詞'
POS2_SENTENCE_FINAL = '終助詞'


def load_j(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as handle:
        return json.load(handle)


AGE_BUCKETS = ["10s", "20s", "30s", "40s", "50s", "60s"]
GENDERS = [("male", "masculine"), ("female", "feminine")]

# 5-tier lexicon (strongly/moderately feminine, neutral, moderately/strongly
# masculine). The weighted roll in GENERATE still only works in 3 categories
# (feminine/neutral/masculine) — SFP_TIERS_BY_CATEGORY collapses the 5 tiers
# back to 3 wherever a flat feminine/neutral/masculine label is needed, while
# the prompt itself walks the full 5-tier fallback chain (see
# sfp_fallback_chain). The DEBUG classifier judges against this same file, so
# generation and classification never disagree about what the tiers mean.
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

# Collapse the 5-tier lexicon down to the 3 categories SFP_CATEGORY_WEIGHTS rolls
# over, and back again: SFP_TIERS_BY_CATEGORY["feminine"] = ["strongly_feminine",
# "moderately_feminine"], etc. "neutral" maps to itself (single tier).
SFP_TIERS_BY_CATEGORY = {
    "feminine": ["strongly_feminine", "moderately_feminine"],
    "neutral": ["neutral"],
    "masculine": ["moderately_masculine", "strongly_masculine"],
}
SFP_CATEGORY_BY_TIER = {
    tier: category for category, tiers in SFP_TIERS_BY_CATEGORY.items() for tier in tiers
}

# Bucket display order/labels/colors for the counter's chart.
BUCKET_DISPLAY_CONFIG = {
    'order': SFP_TIER_LIST,
    'labels': {
        'strongly_masculine': 'Strongly Masculine',
        'moderately_masculine': 'Moderately Masculine',
        'neutral': 'Neutral',
        'moderately_feminine': 'Moderately Feminine',
        'strongly_feminine': 'Strongly Feminine',
    },
    'colors': {
        'Strongly Masculine': '#2166ac',
        'Moderately Masculine': '#67a9cf',
        'Neutral': '#95a5a6',
        'Moderately Feminine': '#ef8a62',
        'Strongly Feminine': '#b2182b',
    },
}
# Display order for the chart legend follows masculine -> neutral -> feminine,
# matching BUCKET_DISPLAY_CONFIG but reusing SFP_TIER_LIST's ordering directly.
BUCKET_DISPLAY_CONFIG['order'] = [
    'strongly_masculine', 'moderately_masculine', 'neutral',
    'moderately_feminine', 'strongly_feminine',
]

# Baseline run (give-the-AI-the-full-list, no weighted roll) is locked — the
# counter always compares the just-generated weight_random run against this
# fixed folder under chat_logs/.
BASELINE_RUN_FOLDER = '20260910_235913'
RUN_DISPLAY_LABELS = {
    'give_all_sfp': 'ベースライン',
    'weight_random': '重み付けランダム',
}

client = None
_client_lock = threading.Lock()


def get_client():
    global client
    if not client:
        with _client_lock:
            if not client:
                if "GENAI_API_KEY" not in os.environ:
                    raise RuntimeError("GENAI_API_KEY is not set")
                client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
    return client


_tagger = None
_tagger_lock = threading.Lock()


def get_tagger():
    global _tagger
    if _tagger is None:
        with _tagger_lock:
            if _tagger is None:
                _tagger = fugashi.Tagger()
    return _tagger


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


def gender_category_from_label(gender_label):
    """Same mapping as get_category, but keyed off the raw 'male'/'female' label
    used in speaker_id/filenames (e.g. parsed from '20s_male') rather than a
    profile dict."""
    if gender_label == "female":
        return "feminine"
    if gender_label == "male":
        return "masculine"
    return "neutral"


def parse_profile(speaker_id: str) -> Dict[str, Optional[str]]:
    """Parse '<age>s_<gender>' (optionally prefixed, e.g. 'preset_a_20s_male')
    speaker ids/filename stems back into {'age': ..., 'gender': ...}."""
    parts = speaker_id.split('_')
    age, gender = None, None
    for p in parts:
        if p.endswith('s') and p[:-1].isdigit():
            age = p
        if p in ('male', 'female'):
            gender = p
    return {'age': age, 'gender': gender}


def make_profile(age, gender_label, gender_category):
    """Build a bare speaker profile with no persona/character data, just age and gender."""
    return {
        "id": f"{age}_{gender_label}",
        "name": f"{age}_{gender_label}",
        "gender_age": f"{age} {gender_label}",
        "gender_category": gender_category,
    }


def all_profiles():
    """Every (age, gender) combination, e.g. 10s male, 10s female, ..., 60s female."""
    return [
        make_profile(age, gender_label, gender_category)
        for age in AGE_BUCKETS
        for gender_label, gender_category in GENDERS
    ]


def all_profile_pairs():
    """Every unordered pairing of profiles, including a profile paired with itself."""
    profiles = all_profiles()
    return [(profiles[i], profiles[j]) for i in range(len(profiles)) for j in range(i, len(profiles))]


# ==============================================================
# GENERATE — persona chat generation with weighted-random SFP steering
# ==============================================================

SFP_CATEGORY_WEIGHTS = {
    # For a speaker of this gender category, probability of rolling
    # masculine / neutral / feminine SFPs for a given turn. Each dict must
    # sum to 1.0.
    "masculine": {"masculine": 0.35, "neutral": 0.50, "feminine": 0.15},
    "feminine": {"masculine": 0.15, "neutral": 0.50, "feminine": 0.35},
    "neutral": {"masculine": 0.0, "neutral": 1.0, "feminine": 0.0},
}

# Second-layer roll: once a turn rolls feminine or masculine (above), this decides
# whether the moderately_* or strongly_* tier is offered as primary. Split per
# category rather than one shared ratio: external data showed strongly_masculine
# usage runs at roughly half strongly_feminine's rate (true even in male-speaker-
# to-male-speaker contexts, so this is a property of the register itself, not just
# of who's listening) — feminine keeps the original 15%, masculine is roughly
# halved to ~8%. Tune these directly if the ratio needs adjusting later.
SFP_INTENSITY_WEIGHTS_BY_CATEGORY = {
    "feminine": {"moderately": 0.85, "strongly": 0.15},
    "masculine": {"moderately": 0.92, "strongly": 0.08},
}

# Toggles for the named anti-attractor bans in build_persona_instruction (below),
# so each can be A/B tested independently against real runs rather than guessing
# which one is behind an observed drift pattern. Both default True (the normal,
# intended behavior) — flip one off here to isolate its effect in a test run.
ENABLE_NANDA_BAN = True
ENABLE_WA_BAN = True  # re-enabled: with it off, strongly_feminine (わ/のよね/かしら
                      # family) ended up dominating 83-93% of all feminine-category
                      # usage for both speakers, nearly all drifted from moderately_
                      # feminine/neutral rolls rather than legitimately rolled —
                      # the earlier "suppresses feminine entirely" risk is being
                      # weighed against this much larger, confirmed drift problem

PIVOT_STRATEGIES = [
    "Ask a follow-up question about a specific detail in what the other person just said.",
    "Share a related but different personal experience of your own, without contrasting it against theirs.",
    "Express a mild difference of opinion or preference, but phrase it in a fresh way rather than a simple 'X is fine, but I prefer Y' contrast.",
    "Agree with or build on a specific detail they mentioned, then add a new detail of your own.",
    "Pivot naturally to a loosely related new topic.",
    "Just react briefly to what they said with a short comment or reaction; no need to pivot or add new information this turn.",
]

RECENT_ENDINGS_KEPT = 3
ENDING_FINGERPRINT_CHARS = 10


def roll_sfp_intensity(rolled_category):
    """Second-layer roll (see SFP_INTENSITY_WEIGHTS_BY_CATEGORY): decides whether
    moderately_* or strongly_* is offered as the primary tier, using the weights
    for whichever side (feminine/masculine) was rolled — the two sides have
    different strongly-rates by design (see that constant's comment), so this
    must be looked up per rolled_category, not from one shared ratio. Callers on
    a neutral roll still call this (and ignore the result) for a stable, position-
    independent random-call sequence; neutral has no entry, so it falls back to
    the masculine weights arbitrarily — harmless since the result goes unused."""
    weights = SFP_INTENSITY_WEIGHTS_BY_CATEGORY.get(rolled_category, SFP_INTENSITY_WEIGHTS_BY_CATEGORY["masculine"])
    intensities = list(weights.keys())
    return random.choices(intensities, weights=[weights[i] for i in intensities], k=1)[0]


def sfp_fallback_chain(rolled_category, rolled_intensity="moderately"):
    """Order the lexicon tiers from 'most aligned with the roll' down to neutral —
    a 'V' shape that always bottoms out at neutral and never crosses to the
    opposite gender's tiers. Previously a feminine/masculine roll's fallback chain
    continued past neutral into the OPPOSITE gender's tiers as a last resort (e.g.
    rolled=feminine could fall all the way to strongly_masculine) — in practice
    this just gave male speakers in particular a path from a neutral/feminine roll
    into masculine-coded particles, defeating the roll's statistical intent. Now:

    rolled=feminine: <rolled_intensity>_feminine -> <other_intensity>_feminine ->
                      neutral -> (no particle; see build_persona_instruction)
    rolled=masculine: mirror image, starting from <rolled_intensity>_masculine.
    rolled=neutral: just neutral -> (no particle) — see build_persona_instruction,
                     which doesn't even call this chain's tail for a neutral roll.
    """
    other_intensity = "strongly" if rolled_intensity == "moderately" else "moderately"

    if rolled_category == "feminine":
        return [f"{rolled_intensity}_feminine", f"{other_intensity}_feminine", "neutral"]
    if rolled_category == "masculine":
        return [f"{rolled_intensity}_masculine", f"{other_intensity}_masculine", "neutral"]
    return ["neutral"]


def find_lockable_form(candidates, reply):
    """Return the first candidate appearing in reply as its own token, not fused into a larger word (e.g. not 私物 for 私)."""
    surfaces = {word.surface for word in get_tagger()(reply)}
    return next((w for w in candidates if w in surfaces), None)


def build_persona_instruction(target, opponent, pronoun_lock=None, koshou_lock=None, recent_endings=None):
    t_cat, o_cat = get_category(target), get_category(opponent)

    # Each turn, roll which pool (masculine / neutral / feminine) supplies the
    # SFP options, weighted by SFP_CATEGORY_WEIGHTS for this speaker's gender
    # category. This gives exact statistical control over masculine/neutral/
    # feminine SFP usage across a sweep, while still presenting the model with
    # a full, natural-sounding list to choose from within the rolled category
    # (no forced single particle).
    weights = SFP_CATEGORY_WEIGHTS.get(t_cat, SFP_CATEGORY_WEIGHTS["neutral"])
    categories = list(weights.keys())
    rolled_cat = random.choices(categories, weights=[weights[c] for c in categories], k=1)[0]

    # Second-layer roll: on a feminine/masculine turn, decide whether moderately_*
    # or strongly_* is offered as primary (see SFP_INTENSITY_WEIGHTS_BY_CATEGORY,
    # per-gender rates) — without this, the fallback chain always led with
    # strongly_*, over-representing it relative to the observed moderately-
    # dominant distribution.
    rolled_intensity = roll_sfp_intensity(rolled_cat)

    # Cap intensity at moderately when the roll crosses to the speaker's OPPOSITE
    # gender category (e.g. a male speaker's t_cat=masculine rolling rolled_cat=
    # feminine, a legitimate ~15% case per SFP_CATEGORY_WEIGHTS). A strongly_*
    # particle from the opposite side read as distinctly less natural for this
    # persona than a moderate one — e.g. a male persona using strongly_feminine
    # わね/だわ stands out far more than moderately_feminine でしょ/もんね does. A
    # speaker's OWN-gender roll (e.g. masculine speaker rolling masculine) is
    # unaffected and can still land on strongly at the normal rate.
    is_opposite_gender_roll = (
        t_cat in ("feminine", "masculine")
        and rolled_cat in ("feminine", "masculine")
        and rolled_cat != t_cat
    )
    if is_opposite_gender_roll and rolled_intensity == "strongly":
        rolled_intensity = "moderately"

    # Walk the lexicon tiers inward toward neutral (see sfp_fallback_chain's
    # docstring — a 'V' shape that never crosses into the opposite gender's
    # tiers), and tell the model to prefer the first tier, only sliding down the
    # chain if every particle in the preferred tier would sound distinctly
    # unnatural; if even neutral doesn't fit, drop the particle rather than
    # reach for any gendered ending at all.
    fallback_tiers = sfp_fallback_chain(rolled_cat, rolled_intensity)
    primary_tier = fallback_tiers[0]
    sfp_pool = TAXONOMIES["sfp"].get(primary_tier, [])
    if sfp_pool:
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
        # Only feminine/masculine rolls have a sibling intensity tier one step down
        # (e.g. primary=moderately_feminine, sibling=strongly_feminine) — a neutral
        # roll's fallback_tiers is just ["neutral"], no sibling to warn about. The
        # two intensities within one gender side often share near-identical surface
        # patterns (moderately_feminine's 動詞の終止形＋の is one character away from
        # strongly_feminine's のね/のよ/のよね), so without calling this out by name
        # the model treats them as interchangeable — this was silently inflating
        # strongly_* usage well past SFP_INTENSITY_WEIGHTS_BY_CATEGORY's intended share.
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
        # んだよ/んだよね/んだ/んだね (all moderately_masculine — the んだ construction adds
        # explanatory nuance, grammatically independent of gender) turned out to be a
        # strong DEFAULT habit for this model: testing showed it reaching for んだよ/
        # んだよね even when the rolled tier was feminine or neutral and that compound
        # was never offered or permitted anywhere in the fallback chain. A generic
        # "don't add だ" rule wasn't specific enough to override this habit, so when
        # the rolled primary tier isn't itself moderately/strongly masculine, name
        # and ban this exact compound outright.
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
        # わ/わね/わよ/わよね/だわ/のよね (all strongly_feminine) turned out to be the
        # mirror-image attractor: testing showed the model defaulting to these
        # canonical "feminine persona voice" particles even when moderately_feminine
        # or neutral was rolled and these forms were never offered or permitted —
        # both retry attempts landing on them back-to-back, so retrying alone
        # couldn't escape it. Same fix as んだ-family: name and ban it outright
        # whenever it isn't actually the rolled tier.
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
    # The "don't end every turn with a question" caveat used to be a blanket rule
    # applied on every single turn — including turns where this same roll picked
    # the "ask a follow-up question" strategy, directly contradicting it. Measured
    # question-mark rate came out near 1% of turns despite that strategy being
    # rolled on ~1/6 of turns (~17%), a ~16x gap — the model was resolving the
    # conflict by suppressing questions almost entirely. Only state the
    # discouragement when this turn's roll isn't the question strategy.
    question_pivot_rolled = pivot_strategy == PIVOT_STRATEGIES[0]
    question_caveat = (
        "" if question_pivot_rolled else
        "Do not end every turn with a question, but you may ask one to naturally explore what the other person just said. "
    )

    if recent_endings:
        endings_list = "、".join(f"「{e}」" for e in recent_endings)
        repetition_inst = f"Avoid ending your reply the same way as these recent endings: {endings_list}. Vary your sentence-final phrasing this turn."
    else:
        repetition_inst = ""

    # SFP/pronoun/address-form guidance is placed right after the persona intro,
    # ahead of the general conversational-style rules below — these are the
    # per-turn constraints we most need the model to actually follow (especially
    # the weighted-roll SFP targeting), and testing across model sizes suggested
    # a smaller/weaker model's prompt-following degrades for instructions buried
    # after a long preamble of less critical stylistic guidance (the classic
    # "lost in the middle" effect, worse on weaker models). The general style
    # rules (pivot strategy, don't-parrot, don't-yes-man, etc.) are important for
    # naturalness but tolerate occasional misses far better than SFP compliance
    # does, so they now come after the hard constraints instead of surrounding them.
    return (
        f"You are a {target['gender_age']}. "
        f"You are talking with a {opponent['gender_age']}."
        "You are having a casual conversation in a room."
        "The topic of the conversation is travel and leisure activities. "
        "Respond naturally in character, as this persona would, based on the conversation so far.\n"
        f"{sfp_inst}\n"
        f"{pron_inst}{kosh_inst}\n"
        "CRITICAL: Do not invent or assume any background details about the other character's situation, schedule, or life. "
        "If the conversation is just starting, introduce yourself naturally and ask a broad, safe icebreaker. "
        "Always respond in Japanese only, regardless of what language the other character uses. "
        "Keep your response short, one or two short sentences at most. "
        f"{question_caveat}"
        "Avoid simply agreeing with or complimenting what the other person said (e.g. do not just say it sounds nice/wonderful/great). " #prevent yes-man responses
        "When you pivot to a new angle or topic, briefly acknowledge what the other person just said first, rather than ignoring it and asserting an unrelated topic in parallel. "
        f"For this turn, react using this approach: {pivot_strategy} "
        "Do not overuse the pattern '[topic] wa ii kedo, watashi/boku wa [other topic]' (e.g. '〜もいいけど、私は〜') — if you used this phrasing recently, use a different approach this turn. " #prevent having the same pivot pattern every turn
        "Every few turns, you may circle back to something mentioned earlier in the conversation instead of only introducing new topics. "
        "Do not repeat or re-name the specific topic word/noun the other person just introduced back to them before reacting (e.g. if they say '動物のやつが好き', don't reply starting with '動物モチーフだと〜'; if they mention '足湯', don't reply starting with '足湯は〜') — just react to the substance directly without restating the keyword. " #prevent parrotting
        f"{repetition_inst}"
        #"When you do share about yourself, link it naturally to what the other character just said. "
    ), rolled_cat, primary_tier, sfp_pool


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


SFP_ENFORCE_MAX_ATTEMPTS = 3


def _classify_reply_sfp(cleaned, turns):
    """Shared by generate_reply's retry loop (every mode) and --mode debug's logging:
    detect the utterance-final SFP (find_final_sfp) and classify it against the
    taxonomy (classify_sfp_with_ai) — same functions the COUNTER section uses on
    saved CSVs, so there is exactly one source of truth for "what SFP did this
    reply use", never a second disagreeing heuristic."""
    sfp = find_final_sfp(cleaned)
    if sfp is None:
        return "", "no_sfp", "no_sfp"
    preceding_context = [f"{t['speaker_name']}: {t['text']}" for t in turns[-3:]]
    marked_text = mark_sfp_in_text(cleaned, sfp['surface'])
    result = classify_sfp_with_ai(TAXONOMIES["sfp"], marked_text, sfp['surface'], preceding_context)
    used_tier = result.get('bucket') or 'no_sfp'
    if used_tier not in SFP_TIER_LIST:
        used_tier = 'no_sfp'
    used_category = SFP_CATEGORY_BY_TIER.get(used_tier, 'no_sfp')
    return sfp['surface'], used_tier, used_category


# だ-copula sentence-final forms (だよ/だね/だよね/だな/だろ/だろう and their ん-contracted
# counterparts んだ/んだよ/んだね/んだよね) are a persistent attractor for this model —
# prompt bans alone (see nanda_control in build_persona_instruction) weren't enough to
# stop it defaulting to these even when neither retry attempt was supposed to use
# them, so _attempt_rank below also penalizes this pattern directly, to at least
# prefer a DIFFERENT wrong-category attempt (if one happened to come up) over a
# だ-family one, since this specific pattern is the one actually driving the
# measured masculine-drift spikes.
DA_FAMILY_ENDINGS = ["んだよね", "んだね", "んだよ", "んだ", "だよね", "だね", "だよ", "だな", "だろう", "だろ"]


def _is_da_family(used_particle):
    particle = used_particle or ""
    return any(particle.endswith(e) for e in DA_FAMILY_ENDINGS)


def _attempt_rank(attempt, rolled_cat):
    """Score one retry attempt for generate_reply's best-of-N fallback (used only
    when no attempt hit an exact tier match — that case exits the loop immediately
    and never needs ranking). Higher is better:
      3 = same category as rolled, just the wrong intensity (e.g. moderately
          rolled, got strongly on the correct side) — closest miss.
      2 = landed on neutral, or no_sfp (a plain ending with no particle at all) —
          when the roll was feminine/masculine, neutral is a much safer miss than
          the OPPOSITE gender (it's not wrong-gendered, just under-shot), so it
          should never be ranked the same as landing on the opposite side.
      0 = the opposite gender entirely (e.g. feminine rolled, landed on
          masculine, or vice versa) — a miss, but at least not the specific
          runaway attractor pattern below.
     -1 = the opposite gender AND specifically a だ-family ending (see
          DA_FAMILY_ENDINGS) — the worst miss. This is ranked below a generic
          opposite-category miss so that if ONE attempt happens to avoid this
          exact known-problem pattern, best-of-N prefers it even when neither
          attempt achieved a real match."""
    if attempt["used_category"] == rolled_cat:
        return 3
    if attempt["used_category"] in ("neutral", "no_sfp"):
        return 2
    if _is_da_family(attempt.get("used_particle")):
        return -1
    return 0


def generate_reply(speaker, opponent, turns, pronoun_lock=None, koshou_lock=None, recent_endings=None, debug_sfp=False):
    system_instruction, rolled_cat, rolled_tier, sfp_pool = build_persona_instruction(
        speaker, opponent, pronoun_lock, koshou_lock, recent_endings)

    # Format transcript from 'speaker's point of view
    history = transcript_text_for_speaker(turns, speaker['id'])

    prompt = (
        f"Conversation history:\n{history}\n\n"
        "It is your turn to speak. Reply naturally in character. "
        "Output ONLY your spoken utterance in Japanese. Do not include your name or any prefixes."
    )

    # Enforce the exact rolled TIER (not just category): classify each attempt's
    # reply and regenerate (up to SFP_ENFORCE_MAX_ATTEMPTS) if it doesn't match,
    # rather than trusting prompt compliance alone — testing showed the model
    # disregards the rolled pool/fallback instructions outright often enough (e.g.
    # landing on masculine particles even when the pool offered was exclusively
    # feminine/neutral) that no further prompt wording closed the gap. Checking
    # used_tier (not just used_category) matters: a moderately_feminine roll and a
    # strongly_feminine result both collapse to the same "feminine" category, so a
    # category-only check was silently accepting the wrong intensity as a "match"
    # on attempt 1 — exactly why strongly_feminine/strongly_masculine kept showing
    # up far more often than SFP_INTENSITY_WEIGHTS_BY_CATEGORY's split should produce.
    # no_sfp (plain ending, no particle at all) counts as an acceptable match for
    # any roll, since a plain ending is never wrong-gendered.
    attempts = []
    overall_start = time.perf_counter()
    for attempt_num in range(1, SFP_ENFORCE_MAX_ATTEMPTS + 1):
        attempt_start = time.perf_counter()
        reply = get_client().models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={"system_instruction": system_instruction},
        ).text.strip()
        cleaned = clean_reply(reply, speaker["name"])
        used_particle, used_tier, used_category = _classify_reply_sfp(cleaned, turns)
        attempt_elapsed = time.perf_counter() - attempt_start

        attempts.append({
            "attempt": attempt_num,
            "cleaned": cleaned,
            "used_particle": used_particle,
            "used_tier": used_tier,
            "used_category": used_category,
            "elapsed": attempt_elapsed,
        })

        if used_tier == rolled_tier or used_category == "no_sfp":
            break

    overall_elapsed = time.perf_counter() - overall_start
    # None of the attempts hit an exact tier match (that would have broken the loop
    # early) — pick the best of what was tried rather than defaulting to whichever
    # happened to run last. Ranked: same category (just wrong intensity, e.g.
    # moderately rolled but got strongly) > no_sfp (a plain ending is never
    # wrong-gendered) > a different category entirely (the worst miss, e.g.
    # feminine rolled but landed on masculine). All attempts already live in the
    # `attempts` list in memory — no persistence/caching needed, picking the best
    # one is just a max() over what's already there.
    final = max(attempts, key=lambda a: _attempt_rank(a, rolled_cat))
    cleaned = final["cleaned"]

    debug_info = None
    if debug_sfp:
        debug_info = {
            "rolled_category": rolled_cat,
            "rolled_tier": rolled_tier,
            "used_particle": final["used_particle"],
            "used_tier": final["used_tier"],
            "used_category": final["used_category"],
            "attempts_used": len(attempts),
            "overall_generation_seconds": round(overall_elapsed, 3),
            "attempt_elapsed_seconds": ";".join(f"{a['elapsed']:.3f}" for a in attempts),
        }
        print(
            f"[SFP DEBUG] speaker={speaker['id']} rolled={rolled_cat} rolled_tier={rolled_tier} "
            f"used_particle={final['used_particle'] or '(none)'} used_tier={final['used_tier']} "
            f"used_category={final['used_category']} attempts={len(attempts)} "
            f"overall_s={overall_elapsed:.3f}"
        )

    return cleaned, overall_elapsed, debug_info


def write_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["turn", "speaker_id", "speaker_name", "listener_id", "listener_name", "text", "generation_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_sfp_debug_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["turn", "speaker_id", "rolled_category", "rolled_tier", "used_particle", "used_tier",
                        "used_category", "attempts_used", "overall_generation_seconds", "attempt_elapsed_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def as_distinct_speakers(speaker_a, speaker_b):
    """Give speaker_a/speaker_b independent instance ids so a self-pairing (e.g. 10s female
    vs 10s female) is tracked as two separate speakers, not one speaker talking to itself."""
    if speaker_a["id"] != speaker_b["id"]:
        return speaker_a, speaker_b
    return (
        {**speaker_a, "id": f"{speaker_a['id']}_a"},
        {**speaker_b, "id": f"{speaker_b['id']}_b"},
    )


def run_chat(speaker_a, speaker_b, turns, output_path, debug_sfp=False):
    speaker_a, speaker_b = as_distinct_speakers(speaker_a, speaker_b)
    transcript = []
    rows = []
    debug_rows = []
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
        reply, elapsed, debug_info = generate_reply(
            speaker, opponent, transcript,
            pronoun_lock=speaker_locks["pronoun"],
            koshou_lock=opponent_locks["koshou"],
            recent_endings=recent_endings[speaker["id"]],
            debug_sfp=debug_sfp,
        )
        if debug_info is not None:
            debug_rows.append({
                "turn": turn_index + 1,
                "speaker_id": speaker["id"],
                **debug_info,
            })

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
    if debug_sfp:
        debug_path = output_path.with_name(f"{output_path.stem}_sfp_debug.csv")
        write_sfp_debug_csv(debug_rows, debug_path)
        print(f"Saved SFP debug log to {debug_path}")
    return output_path


def run_sweep(turns, out_dir, debug_sfp=False):
    """Run every unordered (age, gender) profile pairing, including a profile paired with itself."""
    pairs = all_profile_pairs()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(pairs)
    for count, (a, b) in enumerate(pairs, start=1):
        print(f"[{count}/{total}] {a['gender_age']} vs {b['gender_age']}")
        output_path = out_dir / f"{a['id']}_vs_{b['id']}.csv"
        run_chat(a, b, turns, output_path, debug_sfp=debug_sfp)

    return out_dir


# ==============================================================
# DEBUG — shared SFP classifier (fugashi tagger + Gemini against the taxonomy)
# Used both for live per-turn output above and the COUNTER below, so there is
# exactly one source of truth for "what SFP did this utterance actually use".
# ==============================================================

# Taxonomy compound endings whose LAST token tags as 助動詞 (auxiliary/copula),
# not 終助詞, even though the whole compound functions as a sentence-final form
# in casual speech (んだ ends on だ; だろ/だろう are themselves a single 助動詞
# token) — find_final_sfp's run-acceptance check below needs this list to accept
# such a run, since checking the last token's POS tag alone would reject them.
# わな is here too for a different reason: fugashi mis-tags it as the common noun
# 輪/輪っか, so it never even enters the 助詞/助動詞 run — it's matched directly
# against the raw utterance ending instead (see the final fallback loop).
AUXILIARY_SFP_ENDINGS = ["んだ", "だろう", "だろ"]
COMPOUND_SFP_FALLBACK_ENDINGS = ["わな"]

# POS categories that can appear inside a trailing sentence-final particle run:
# 助詞 (particle, any subtype — 終助詞, 準体助詞, etc.) and 助動詞 (auxiliary/copula,
# covers だ/です/だろう and similar). A content word (noun/verb/adjective) or
# punctuation ends the run.
PARTICLE_RUN_POS1 = ('助詞', '助動詞')


def find_final_sfp(text: str) -> Optional[Dict[str, str]]:
    """Tokenize one utterance and return its trailing run of particle/auxiliary
    tokens if that run is a genuine sentence-final form, ignoring trailing
    punctuation/symbols.

    A compound ending like んだよね is several tokens (ん/準体助詞 + だ/助動詞 +
    よ/終助詞 + ね/終助詞) — returning only the last one (ね) truncates the real
    ending to a single character and feeds the classifier the wrong, narrower
    unit to judge, which is what caused used_particle to show just 'ね' for a
    sentence actually ending in んだよね, and caused the same compound to be
    classified inconsistently from one occurrence to the next (the classifier
    was judging bare ね sometimes, and getting lucky with fuller context other
    times). Walking backward through the whole contiguous 助詞/助動詞 run fixes
    this: the full compound is captured as one unit.

    The run is accepted if its last token is a genuine 終助詞 (the common case:
    よね, だね, かよ, ...), OR if the run's surface form ends in one of
    AUXILIARY_SFP_ENDINGS (んだ, だろ, だろう — real taxonomy entries whose final
    token tags as 助動詞 rather than 終助詞). Either way, a multi-clause utterance
    whose last clause ends on a non-particle (e.g. '...なんだよね。得意なほう？' —
    true end is 方(ほう), a noun) still correctly returns None, since ほう isn't
    助詞/助動詞 at all and so never enters the run.

    Falls back to COMPOUND_SFP_FALLBACK_ENDINGS for the one compound taxonomy
    entry (わな) the POS tagger mis-tags as an unrelated noun rather than
    particle/auxiliary tokens — this fallback only matches the literal end of
    the utterance, so it can't reintroduce the multi-clause bug above."""
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
            lemma = ''.join(w.feature.lemma or w.surface for w in run)
            return {'surface': surface, 'lemma': lemma}

    stripped = text.rstrip("。！？!?　 ")
    for ending in COMPOUND_SFP_FALLBACK_ENDINGS:
        if stripped.endswith(ending):
            return {'surface': ending, 'lemma': ending}
    return None


TRAILING_PUNCTUATION = "。！？!?　 "


def mark_sfp_in_text(text: str, surface: str) -> str:
    """Wrap `surface` (find_final_sfp's detected ending) in 【】 at its actual
    position in `text`, tolerating trailing punctuation after it (。！？ etc.) —
    text almost always ends in surface + punctuation (e.g. '...んだよね。'), not
    surface alone, so a naive text.endswith(surface) check fails and silently
    falls through to appending the bracket after the punctuation instead of
    replacing the real ending (e.g. '...んだよね。【んだよね】', doubling the
    particle and marking nothing useful for the classifier to judge)."""
    stripped = text.rstrip(TRAILING_PUNCTUATION)
    trailing_punct = text[len(stripped):]
    if stripped.endswith(surface):
        return stripped[: -len(surface)] + f"【{surface}】" + trailing_punct
    return text + f"【{surface}】"


def build_classification_prompt(
    taxonomy: Dict[str, List[str]], utterance_text: str, marked_particle: str,
    preceding_context: List[str],
) -> str:
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


def classify_sfp_with_ai(
    taxonomy: Dict[str, List[str]], utterance_text: str, marked_particle: str,
    preceding_context: List[str],
) -> Dict[str, Any]:
    prompt = build_classification_prompt(taxonomy, utterance_text, marked_particle, preceding_context)
    response = get_client().models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0, "seed": 0},
    ).text.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {'bucket': None, 'matched_entry': None}


# ==============================================================
# COUNTER — pool saved chat_logs CSVs across runs, classify every
# utterance-final SFP with the DEBUG classifier above, and chart the
# resulting gender-bucket distribution per run/speaker.
# ==============================================================

def classify_conversation_file(
    filepath: Path, run_label: str, taxonomy: Dict[str, List[str]],
    result_dir: Path, context_window: int = 3, max_workers: int = AI_CLASSIFY_MAX_WORKERS,
    use_cache: bool = True,
) -> pd.DataFrame:
    # Per-file results are kept under result_dir/per_file/<run_label>/<source folder>/
    # (keyed by source folder name, not just run_label, so pointing a run label at a
    # different chat_logs/ folder can't be confused with another folder's results)
    # so an interrupted run can be resumed without re-classifying finished files.
    file_result_path = result_dir / 'per_file' / run_label / filepath.parent.name / f'{filepath.stem}.csv'
    if use_cache and file_result_path.exists():
        return pd.read_csv(file_result_path)

    df = pd.read_csv(filepath)
    if df.empty or 'text' not in df.columns:
        return pd.DataFrame()

    rows = df.to_dict('records')

    candidates = []
    for i, row in enumerate(rows):
        sfp = find_final_sfp(row.get('text', ''))
        if sfp is None:
            continue
        surface = sfp['surface']
        marked_text = mark_sfp_in_text(str(row['text']), surface)
        preceding_context = [
            f"{rows[j]['speaker_name']}: {rows[j]['text']}"
            for j in range(max(0, i - context_window), i)
        ]
        candidates.append((i, row, sfp, marked_text, preceding_context))

    if not candidates:
        return pd.DataFrame()

    results = [None] * len(candidates)

    def classify_one(idx):
        _, row, sfp, marked_text, preceding_context = candidates[idx]
        result = classify_sfp_with_ai(taxonomy, marked_text, sfp['surface'], preceding_context)
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_one, idx): idx for idx in range(len(candidates))}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    out_rows = []
    for (i, row, sfp, marked_text, _), result in zip(candidates, results):
        profile = parse_profile(str(row.get('speaker_id', '')))
        out_rows.append({
            'run': run_label,
            'file': filepath.name,
            'turn': row.get('turn'),
            'speaker_id': row.get('speaker_id'),
            'speaker_age': profile['age'],
            'speaker_gender': profile['gender'],
            'speaker_gender_category': gender_category_from_label(profile['gender']),
            'listener_id': row.get('listener_id'),
            'text': row.get('text'),
            'marked_particle': sfp['surface'],
            'ai_bucket': result.get('bucket'),
            'ai_matched_entry': result.get('matched_entry'),
        })

    df_result = pd.DataFrame(out_rows)
    if not df_result.empty:
        file_result_path.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(file_result_path, index=False, encoding='utf-8-sig')
    return df_result


def file_matches_pair(filepath: Path, pair: tuple) -> bool:
    """True if the two speaker profiles ('20s', 'male') / ('20s', 'female') both
    appear in the filename (e.g. '20s_male_vs_20s_female.csv' or the
    'preset_a_20s_male_vs_preset_b_20s_female.csv' naming), regardless of order."""
    stem = filepath.stem
    wanted = {f"{age}_{gender}" for age, gender in pair}
    found = {f"{p['age']}_{p['gender']}" for part in stem.split('_vs_')
             if (p := parse_profile(part))['age'] and p['gender']}
    return wanted == found


def classify_run(run_label: str, run_dir: Path, taxonomy: Dict[str, List[str]],
                  result_dir: Path, context_window: int = 3, use_cache: bool = True,
                  pair_filter: Optional[tuple] = None) -> pd.DataFrame:
    files = sorted(run_dir.glob('*.csv'))
    files = [f for f in files if not f.stem.endswith('_sfp_debug')]
    if pair_filter:
        files = [f for f in files if file_matches_pair(f, pair_filter)]
    if not files:
        print(f"Warning: no CSV files found under {run_dir}"
              + (f" matching pair {pair_filter}" if pair_filter else ""))
        return pd.DataFrame()

    all_dfs = []
    for i, filepath in enumerate(files, start=1):
        print(f"[{run_label}] [{i}/{len(files)}] {filepath.name}")
        df = classify_conversation_file(
            filepath, run_label, taxonomy, result_dir,
            context_window=context_window, use_cache=use_cache)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def build_bucket_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket distribution per (run, speaker), so each speaker in a conversation
    (e.g. the 20s_male side vs. the 20s_female side) is reported separately."""
    config = BUCKET_DISPLAY_CONFIG
    df = df[df['ai_bucket'].notna()]
    group_cols = ['run', 'speaker_id']
    dist = (
        df.groupby(group_cols)['ai_bucket']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0) * 100
    ).round(2)
    dist = dist.reindex(columns=config['order'], fill_value=0.0)
    dist = dist.rename(columns=config['labels'])
    dist['n_sfp'] = df.groupby(group_cols).size()
    return dist


def plot_bucket_distribution(bucket_dist: pd.DataFrame, output_path: Path):
    """Stacked horizontal bar chart: % gender-bucket SFP usage per (run, speaker),
    so the two generation methods can be visually compared."""
    if bucket_dist.empty:
        print("No bucket distribution to plot.")
        return

    config = BUCKET_DISPLAY_CONFIG
    bucket_labels = [config['labels'][k] for k in config['order']]

    labels = [
        f"{RUN_DISPLAY_LABELS.get(run, run)}\n{speaker}"
        for run, speaker in bucket_dist.index
    ]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(labels))))

    left = pd.Series(0.0, index=bucket_dist.index)
    for bucket in bucket_labels:
        values = bucket_dist[bucket]
        bars = ax.barh(labels, values, left=left, label=bucket, color=config['colors'][bucket])
        for bar, pct in zip(bars, values):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        f'{pct:.1f}%', ha='center', va='center', fontsize=8, color='white')
        left += values

    ax.set_xlabel('Percentage of classified SFPs (%)')
    ax.set_xlim(0, 100)
    ax.set_title('SFP Gender-Bucket Usage by Run and Speaker', pad=40)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(bucket_labels), frameon=False)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved chart to: {output_path}")


def run_counter(runs: Dict[str, str], context_window: int = 3, use_cache: bool = True,
                 pair_filter: Optional[tuple] = None) -> Optional[Path]:
    """Classify every run in `runs` ({label: chat_logs subfolder name}), pool the
    results, and save the bucket-distribution CSV + chart under a fresh timestamped
    result_dir. Returns the result_dir, or None if nothing was classified."""
    taxonomy = TAXONOMIES["sfp"]
    result_dir = COUNTER_OUTPUT_DIR / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COUNTER — SFP gender-bucket naturalness cross-check")
    print(f"Runs: {runs}")
    print(f"Output: {result_dir}")
    print("=" * 60)

    all_dfs = []
    for label, folder in runs.items():
        run_dir = CHAT_LOGS_DIR / folder
        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping '{label}'")
            continue
        df_run = classify_run(
            label, run_dir, taxonomy, result_dir,
            context_window=context_window, use_cache=use_cache,
            pair_filter=pair_filter,
        )
        if not df_run.empty:
            all_dfs.append(df_run)

    if not all_dfs:
        print("No SFPs classified — nothing to summarize.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    detailed_csv = result_dir / 'sfp_classification_detailed.csv'
    df_all.to_csv(detailed_csv, index=False, encoding='utf-8-sig')
    print(f"\nSaved detailed classification to: {detailed_csv}")

    bucket_dist = build_bucket_distribution(df_all)
    print("\nSFP gender-bucket distribution by run (%):")
    print(bucket_dist.to_string())
    bucket_dist.to_csv(result_dir / 'sfp_bucket_distribution_by_run.csv', encoding='utf-8-sig')

    plot_bucket_distribution(bucket_dist, result_dir / 'sfp_bucket_distribution.svg')
    return result_dir


# ==============================================================
# CLI
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM_Response weighted-random SFP persona chat generator + counter. "
            "--mode generate/debug drive a chat (single pairing via --age-a/--gender-a/"
            "--age-b/--gender-b, or a full sweep if omitted), then automatically run the "
            "counter step comparing this run ('weight_random') against the locked "
            "'give_all_sfp' baseline. --mode counter only re-runs the comparison/chart "
            "over existing chat_logs folders."
        )
    )
    parser.add_argument("--mode", choices=["generate", "debug", "counter"], default="generate",
                         help="generate = chat only. debug = chat + live per-turn SFP classification "
                              "(prints a debug line, saves a sibling *_sfp_debug.csv). counter = skip "
                              "generation, just classify+chart existing chat_logs folders (see --runs).")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS, help="Total number of utterances to generate per pairing")
    parser.add_argument("--age-a", choices=AGE_BUCKETS, help="Age bucket for speaker A (requires --gender-a/--age-b/--gender-b)")
    parser.add_argument("--gender-a", choices=["male", "female"], help="Gender for speaker A")
    parser.add_argument("--age-b", choices=AGE_BUCKETS, help="Age bucket for speaker B")
    parser.add_argument("--gender-b", choices=["male", "female"], help="Gender for speaker B")
    parser.add_argument(
        "--runs", nargs='+', default=None,
        help=(
            "--mode counter only: space-separated \"label=folder_name\" pairs under chat_logs/ "
            "to compare (default: give_all_sfp=<locked baseline> weight_random=<most recent run>)."
        ),
    )
    parser.add_argument("--context-window", type=int, default=3,
                         help="Number of preceding turns shown to the classifier as context.")
    parser.add_argument("--no-cache", action="store_true",
                         help="Ignore cached per-file classifications and re-classify everything.")
    args = parser.parse_args()

    if args.mode == "counter":
        if args.runs:
            runs = {}
            for item in args.runs:
                label, _, folder = item.partition('=')
                runs[label] = folder
        else:
            latest_run = max(CHAT_LOGS_DIR.iterdir(), key=lambda p: p.stat().st_mtime).name
            runs = {"give_all_sfp": BASELINE_RUN_FOLDER, "weight_random": latest_run}
        run_counter(runs, context_window=args.context_window, use_cache=not args.no_cache)
        return

    single_pair_args = (args.age_a, args.gender_a, args.age_b, args.gender_b)
    if any(single_pair_args) and not all(single_pair_args):
        parser.error("--age-a/--gender-a/--age-b/--gender-b must all be given together")

    debug_sfp = args.mode == "debug"
    gender_category = {"male": "masculine", "female": "feminine"}

    if all(single_pair_args):
        profile_a = make_profile(args.age_a, args.gender_a, gender_category[args.gender_a])
        profile_b = make_profile(args.age_b, args.gender_b, gender_category[args.gender_b])
        run_dir = OUTPUTS_DIR / "chat_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / f"{profile_a['id']}_vs_{profile_b['id']}.csv"
        print(f"Running {profile_a['gender_age']} vs {profile_b['gender_age']}, {args.turns} turns...")
        result = run_chat(profile_a, profile_b, args.turns, output_path, debug_sfp=debug_sfp)
        print(f"Saved chat log to {result}")
    else:
        run_dir = OUTPUTS_DIR / "chat_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Sweeping all age/gender pairings, {args.turns} turns each...")
        run_sweep(args.turns, run_dir, debug_sfp=debug_sfp)
        print(f"Saved sweep chat logs to {run_dir}")

    # Always chain into the counter: compare this run (as 'weight_random') against
    # the locked give_all_sfp baseline.
    run_counter(
        {"give_all_sfp": BASELINE_RUN_FOLDER, "weight_random": run_dir.name},
        context_window=args.context_window,
    )


if __name__ == '__main__':
    main()
