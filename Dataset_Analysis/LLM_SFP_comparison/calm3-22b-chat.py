"""calm3-22b-chat style conversion and sentence-final-particle analysis.

What this script provides:
1. Load 1-to-1 CEJC conversation metadata from Conversation.csv.
2. Read LUU CSV (長単位発話) for utterance text — one LUU line per LLM call.
3. Read morphSUW CSV for pre-conversion sentence-final-particle counts only.
4. Convert utterances to a target age/gender style using calm3-22b-chat.
5. Count sentence-final particles before/after conversion, focused on: ね / よ / よね.

Note:
- "before" counts use POS tags from morphSUW (助詞-終助詞), linked to LUU via
  (speakerID, startTime, endTime) ↔ (話者ラベル, 発話単位の開始時刻, 発話単位の終了時刻).
- "after" counts are estimated from generated plain text via regex heuristics.
- Utterances that are purely annotation tags or short aizuchi are passed through
  unchanged without calling the LLM.
- calm3-22b-chat uses a chat-template with system + user turns; few-shot examples
  are embedded in the user turn to anchor the model on the correct output style.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


MODEL_ID_DEFAULT = "cyberagent/calm3-22b-chat"
TARGET_PARTICLES = ("ね", "よ", "よね")


@dataclass(frozen=True)
class FixedRunConfig:
    """Edit this block to run a fixed CEJC conversion without CLI flags."""

    conversation_dir: Path
    target_age: str
    target_gender: str
    max_utterances: Optional[int] = None


FIXED_RUN_CONFIG = FixedRunConfig(
    conversation_dir=Path("C001/C001_002"),
    target_age="50-54",
    target_gender="male",
    max_utterances=None,
)

# ── Annotation-tag regexes (LUU inline format) ───────────────────────────────
# Matches CEJC inline tags: (L), (D ...), (F ...), (R ...), (U ...), (W ...), (X ...)
INLINE_TAGS = re.compile(r"\([A-Za-z][^)]*\)")

# Matches pause duration annotations, e.g. (0.459) or (0.13)
PAUSE_ANNOTATIONS = re.compile(r"\(\d+\.\d+\)")

# After stripping tags, nothing meaningful remains → safe to pass through.
WHOLLY_ANNOTATION = re.compile(r"^[\s。、！？：:]*$")

# Short backchannel surfaces to pass through even without wrapper tags.
AIZUCHI_SURFACES = frozenset({"うん", "ああ", "あー", "ね", "え", "◇", "あ", "うーん", "はい"})

# ── 4-shot examples derived from C001_002 → 50s Male conversion ──────────────
# Each pair is (cleaned_original, target_style_output).
# Chosen to show: greeting shift, particle addition, vocabulary change, longer utterance.
FEWSHOT_EXAMPLES: List[tuple[str, str]] = [
    ("おはようございます。",                                             "おう、おはよ。"),
    ("喉がらがら。",                                                     "喉がらがらなんだよ"),
    ("超痛い。",                                                         "めちゃくちゃ痛い"),
    ("最近なんかえーと自転車用にケイデンス計りたくて回転数とか。",           "最近まあえーと自転車用にケイデンス計りたくて回転数とか"),
]


def should_passthrough(text: str) -> bool:
    """Return True if this utterance should be copied as-is without LLM conversion."""
    stripped = text.strip()
    content_only = INLINE_TAGS.sub("", stripped)
    content_only = PAUSE_ANNOTATIONS.sub("", content_only).strip().rstrip("。、！？：:")

    if WHOLLY_ANNOTATION.fullmatch(content_only):
        return True
    if len(content_only) <= 3 and content_only in AIZUCHI_SURFACES:
        return True
    return False


def clean_for_llm(text: str) -> str:
    """Strip CEJC annotation tags, pause annotations, and normalise whitespace."""
    cleaned = INLINE_TAGS.sub("", text)
    cleaned = PAUSE_ANNOTATIONS.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_age_label(age: str) -> str:
    """Append 歳 when missing (e.g. '20-24' → '20-24歳')."""
    value = age.strip()
    if value and "歳" not in value:
        return f"{value}歳"
    return value


def normalize_gender_label(gender: str) -> str:
    """Accept English gender inputs and convert to Japanese labels."""
    lookup = {
        "female": "女性", "f": "女性", "woman": "女性",
        "male":   "男性", "m": "男性", "man":   "男性",
        "女性":   "女性", "男性": "男性",
    }
    return lookup.get(gender.strip().lower(), gender.strip())


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class SpeakerProfile:
    speaker_id: str
    age: str
    gender: str


@dataclass
class Utterance:
    speaker_id: str
    text: str          # raw LUU text, preserved in original_text column
    token_surfaces: List[str]
    sfp_tokens: List[str]


# ── XML helpers ───────────────────────────────────────────────────────────────

def load_one_to_one_conversation_ids(conversations_xml: Path) -> set[str]:
    """Return CEJC conversation IDs where 話者数 == 2."""
    tree = ET.parse(conversations_xml)
    root = tree.getroot()
    one_to_one = set()
    for conv in root.findall("conversation"):
        conv_id = (conv.findtext("会話ID") or "").strip()
        n_speakers = (conv.findtext("話者数") or "").strip()
        if conv_id and n_speakers == "2":
            one_to_one.add(conv_id)
    return one_to_one


def load_speaker_profiles(informants_xml: Path) -> Dict[str, SpeakerProfile]:
    """Load speaker metadata from informants.xml."""
    tree = ET.parse(informants_xml)
    root = tree.getroot()
    profiles: Dict[str, SpeakerProfile] = {}
    for inf in root.findall("informant"):
        speaker_id = (inf.findtext("話者ID") or "").strip()
        if not speaker_id:
            continue
        profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            age=(inf.findtext("年齢") or "NA").strip(),
            gender=(inf.findtext("性別") or "NA").strip(),
        )
    return profiles


def parse_utterance_from_u_element(u_elem: ET.Element) -> Utterance:
    """Extract utterance text and SFP tokens from one <u> node."""
    speaker_id = (u_elem.attrib.get("speakerID") or "").strip()
    token_surfaces: List[str] = []
    sfp_tokens: List[str] = []
    for l_elem in u_elem.findall("l"):
        pos = l_elem.attrib.get("p", "")
        token = (l_elem.attrib.get("t") or "").strip()
        if token and not token.startswith("himawari_"):
            token_surfaces.append(token)
        if pos == "助詞-終助詞" and token:
            sfp_tokens.append(token)
    text = "".join(token_surfaces)
    return Utterance(speaker_id=speaker_id, text=text,
                     token_surfaces=token_surfaces, sfp_tokens=sfp_tokens)


def extract_dialogue_history_from_corpus(corpus_xml: Path) -> tuple[str, List[Utterance]]:
    """Read a UTF-16 Himawari corpus XML and return (conversation_id, utterances)."""
    conv_id = ""
    history: List[Utterance] = []
    for event, elem in ET.iterparse(corpus_xml, events=("start", "end")):
        if event == "start" and elem.tag == "cejc":
            conv_id = (elem.attrib.get("name") or "").strip()
        if event == "end" and elem.tag == "u":
            history.append(parse_utterance_from_u_element(elem))
            elem.clear()
    return conv_id, history


# ── Particle counting ─────────────────────────────────────────────────────────

def count_particles_from_sfp_tokens(sfp_tokens: List[str]) -> Dict[str, int]:
    """Count ね/よ/よね from morphSUW POS-tagged token list."""
    counts = {"ね": 0, "よ": 0, "よね": 0}
    i = 0
    while i < len(sfp_tokens):
        if i + 1 < len(sfp_tokens) and sfp_tokens[i] == "よ" and sfp_tokens[i + 1] == "ね":
            counts["よね"] += 1
            i += 2
            continue
        if sfp_tokens[i] == "ね":
            counts["ね"] += 1
        elif sfp_tokens[i] == "よ":
            counts["よ"] += 1
        i += 1
    return counts


def count_particles_from_plain_text(text: str) -> Dict[str, int]:
    """Heuristic count of sentence-final particles in generated plain text."""
    counts = {"ね": 0, "よ": 0, "よね": 0}
    yone_matches = re.findall(r"よね(?=[。！？!?\n]|$)", text)
    counts["よね"] = len(yone_matches)
    text_wo_yone = re.sub(r"よね(?=[。！？!?\n]|$)", "", text)
    counts["よ"] = len(re.findall(r"よ(?=[。！？!?\n]|$)", text_wo_yone))
    counts["ね"] = len(re.findall(r"ね(?=[。！？!?\n]|$)", text_wo_yone))
    return counts


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_chat_messages(
    input_text: str,
    history_text: str,
    source_age: str,
    source_gender: str,
    target_age: str,
    target_gender: str,
    examples: List[tuple[str, str]] = FEWSHOT_EXAMPLES,
) -> List[dict]:
    """Build system+user chat messages for calm3-22b-chat.

    Key design decisions (ported from Swallow's build_chat_prompt):
    - System prompt is minimal: one-line role declaration only.
    - Few-shot examples live in the user turn so the model sees the exact
      input→output pattern rather than abstract rules.
    - The user turn ends with 「発話: {input}\n変換:」to strongly prime the
      model to complete only the converted utterance.
    - Conversation history (cleaned of annotation tags) is included so the
      model has discourse context, but is clearly separated from the task.
    """
    system_prompt = (
        "あなたは日本語のスタイル変換アシスタントです。"
        "変換後の発話本文のみを1文で出力してください。説明・コメント不要。"
    )

    shot_block = "\n".join(
        f"発話: {src}\n変換: {tgt}" for src, tgt in examples
    )
    context_block = f"会話履歴:\n{history_text}\n\n" if history_text.strip() else ""

    user_content = (
        "以下の変換例を参考に、最後の発話を目標話者スタイルに変換してください。\n"
        "変換後の発話本文のみ出力してください。説明・コメント不要。\n\n"
        f"入力話者: 年齢={source_age} 性別={source_gender}\n"
        f"目標話者: 年齢={target_age} 性別={target_gender}\n\n"
        f"【変換例】\n{shot_block}\n\n"
        f"{context_block}"
        f"発話: {input_text}\n変換:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


# ── LLM style converter ───────────────────────────────────────────────────────

class Calm3StyleConverter:
    def __init__(self, model_id: str = MODEL_ID_DEFAULT, max_new_tokens: int = 64):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "transformers/torch が必要です。"
                "例: pip install transformers torch accelerate"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"[Calm3StyleConverter] Loaded '{self.model_id}'.")

    def convert(
        self,
        input_text: str,
        conversation_history: Iterable[Utterance],
        source_age: str,
        source_gender: str,
        target_age: str,
        target_gender: str,
    ) -> str:
        """Convert one utterance's style to the target age/gender."""
        self._lazy_load()
        assert self._tokenizer is not None
        assert self._model is not None

        # Clean annotation tags from history (same as Swallow)
        history_text = "\n".join(
            f"{u.speaker_id}: {clean_for_llm(u.text)}"
            for u in list(conversation_history)[-8:]
            if clean_for_llm(u.text).strip()
        )

        messages = build_chat_messages(
            input_text, history_text,
            source_age, source_gender, target_age, target_gender,
        )
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        # Keep only the first non-empty line — prevents the model from
        # continuing with the next 発話: line of the few-shot pattern.
        first_line = generated.split("\n")[0].strip()
        return first_line if first_line else generated.strip()


# ── CSV loaders ───────────────────────────────────────────────────────────────

def _normalize_ts(ts: str) -> float:
    return round(float(ts), 3)


def load_cejc_one_to_one_ids(conversation_csv: Path) -> List[str]:
    ids: List[str] = []
    with conversation_csv.open("r", encoding="shift_jis", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("話者数") or "").strip() == "2":
                conv_id = (row.get("会話ID") or "").strip()
                if conv_id:
                    ids.append(conv_id)
    return sorted(set(ids))


def resolve_cejc_paths(cejc_root: Path, conversation_id: str) -> tuple[Path, Path]:
    """Return (luu_path, morph_path) for the given conversation ID."""
    session_id = conversation_id.split("_")[0]
    conv_dir = cejc_root / session_id / conversation_id
    luu_path = conv_dir / f"{conversation_id}-luu.csv"
    morph_path = conv_dir / f"{conversation_id}-morphSUW.csv"
    if not luu_path.exists() or not morph_path.exists():
        raise FileNotFoundError(f"Required files not found under {conv_dir}")
    return luu_path, morph_path


def conversation_id_from_directory(cejc_root: Path, conversation_dir: Path) -> str:
    """Resolve conversation ID from a directory path and validate required files."""
    conv_dir = conversation_dir if conversation_dir.is_absolute() else cejc_root / conversation_dir
    if not conv_dir.exists() or not conv_dir.is_dir():
        raise FileNotFoundError(f"Conversation directory not found: {conv_dir}")
    conversation_id = conv_dir.name
    luu_path = conv_dir / f"{conversation_id}-luu.csv"
    morph_path = conv_dir / f"{conversation_id}-morphSUW.csv"
    if not luu_path.exists() or not morph_path.exists():
        raise FileNotFoundError(
            f"Required files missing: {luu_path.name}, {morph_path.name}"
        )
    return conversation_id


def load_speaker_data(speaker_data_csv: Path) -> Dict[str, SpeakerProfile]:
    profiles: Dict[str, SpeakerProfile] = {}
    with speaker_data_csv.open("r", encoding="shift_jis", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_id = (row.get("話者ID") or "").strip()
            if not speaker_id:
                continue
            profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                age=(row.get("年齢") or "NA").strip(),
                gender=(row.get("性別") or "NA").strip(),
            )
    return profiles


def load_label_to_speaker_id_map(relation_csv: Path, conversation_id: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with relation_csv.open("r", encoding="shift_jis", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("会話ID") or "").strip() != conversation_id:
                continue
            label = (row.get("話者ラベル") or "").strip()
            speaker_id = (row.get("話者ID") or "").strip()
            if label and speaker_id:
                mapping[label] = speaker_id
    return mapping


def load_utterances_from_luu_csv(luu_path: Path, morph_path: Path) -> List[Utterance]:
    """Load utterances from LUU CSV, attaching SFP tokens from morphSUW.

    One LUU row = one utterance fed to the LLM.
    Linking key: (speakerID, startTime, endTime) ↔
                 (話者ラベル, 発話単位の開始時刻, 発話単位の終了時刻).
    """
    sfp_by_utt: Dict[tuple[str, float, float], List[str]] = {}
    with morph_path.open("r", encoding="shift_jis", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("品詞") or "").strip() != "助詞-終助詞":
                continue
            speaker = (row.get("話者ラベル") or "").strip()
            start = _normalize_ts((row.get("発話単位の開始時刻") or "0").strip())
            end = _normalize_ts((row.get("発話単位の終了時刻") or "0").strip())
            token = (row.get("語彙素") or row.get("書字形") or "").strip()
            if not token:
                continue
            sfp_by_utt.setdefault((speaker, start, end), []).append(token)

    history: List[Utterance] = []
    with luu_path.open("r", encoding="shift_jis", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = (row.get("speakerID") or "").strip()
            text = (row.get("text") or "").strip()
            if not speaker or not text:
                continue
            start = _normalize_ts((row.get("startTime") or "0").strip())
            end = _normalize_ts((row.get("endTime") or "0").strip())
            sfp_tokens = sfp_by_utt.get((speaker, start, end), [])
            history.append(Utterance(
                speaker_id=speaker, text=text,
                token_surfaces=[], sfp_tokens=sfp_tokens,
            ))
    return history


# ── Main analysis ─────────────────────────────────────────────────────────────

def choose_conversation_id(one_to_one_ids: List[str], provided_id: Optional[str]) -> str:
    if provided_id:
        if provided_id not in one_to_one_ids:
            raise ValueError(f"{provided_id} is not a 1-to-1 conversation ID.")
        return provided_id

    print("Available 1-to-1 conversations (first 40):")
    for conv_id in one_to_one_ids[:40]:
        print(f"- {conv_id}")
    if len(one_to_one_ids) > 40:
        print(f"... and {len(one_to_one_ids) - 40} more")

    selected = input("Which conversation ID to convert? (e.g., C001_002): ").strip()
    if selected not in one_to_one_ids:
        raise ValueError("Invalid conversation ID or not 1-to-1.")
    return selected


def run_selected_conversation_analysis(
    conversation_id: str,
    cejc_root: Path,
    speaker_data_csv: Path,
    relation_csv: Path,
    target_age: str,
    target_gender: str,
    output_csv: Path,
    model_id: str = MODEL_ID_DEFAULT,
    max_utterances: Optional[int] = None,
) -> None:
    luu_path, morph_path = resolve_cejc_paths(cejc_root, conversation_id)
    history = load_utterances_from_luu_csv(luu_path, morph_path)
    if max_utterances is not None:
        history = history[:max_utterances]

    profiles = load_speaker_data(speaker_data_csv)
    label_to_speaker_id = load_label_to_speaker_id_map(relation_csv, conversation_id)
    converter = Calm3StyleConverter(model_id=model_id)

    print("Source speaker metadata:")
    for speaker_label in sorted(set(u.speaker_id for u in history)):
        sid = label_to_speaker_id.get(speaker_label, "NA")
        profile = profiles.get(sid, SpeakerProfile(sid, "NA", "NA"))
        print(
            f"- {speaker_label} -> speaker_id={sid}, "
            f"source_age={profile.age}, source_gender={profile.gender}"
        )
    print(f"Target style: target_age={target_age}, target_gender={target_gender}")

    rows: List[Dict[str, str]] = []
    for idx, utt in enumerate(history):
        sid = label_to_speaker_id.get(utt.speaker_id, "NA")
        profile = profiles.get(sid, SpeakerProfile(sid, "NA", "NA"))

        if should_passthrough(utt.text):
            converted = utt.text
        else:
            cleaned_input = clean_for_llm(utt.text)
            converted = converter.convert(
                input_text=cleaned_input,
                conversation_history=history[max(0, idx - 8): idx],
                source_age=profile.age,
                source_gender=profile.gender,
                target_age=target_age,
                target_gender=target_gender,
            )

        pre_counts = count_particles_from_sfp_tokens(utt.sfp_tokens)
        post_counts = count_particles_from_plain_text(converted)

        rows.append({
            "conversation_id": conversation_id,
            "speaker_label":   utt.speaker_id,
            "speaker_id":      sid,
            "source_age":      profile.age,
            "source_gender":   profile.gender,
            "target_age":      target_age,
            "target_gender":   target_gender,
            "original_text":   utt.text,
            "converted_text":  converted,
            "pre_ね":  str(pre_counts["ね"]),
            "pre_よ":  str(pre_counts["よ"]),
            "pre_よね": str(pre_counts["よね"]),
            "post_ね": str(post_counts["ね"]),
            "post_よ": str(post_counts["よ"]),
            "post_よね": str(post_counts["よね"]),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "conversation_id", "speaker_label", "speaker_id",
        "source_age", "source_gender", "target_age", "target_gender",
        "original_text", "converted_text",
        "pre_ね", "pre_よ", "pre_よね",
        "post_ね", "post_よ", "post_よね",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Style conversion with calm3-22b-chat using CEJC LUU data"
    )
    script_dir = Path(__file__).parent.parent
    cejc_dir = script_dir / "CEJC"
    output_dir = Path(__file__).parent

    parser.add_argument("--cejc-root", type=Path, default=cejc_dir)
    parser.add_argument("--conversation-csv", type=Path,
                        default=cejc_dir / "Conversation.csv")
    parser.add_argument("--relation-csv", type=Path,
                        default=cejc_dir / "Speaker_Conversation_Relation.csv")
    parser.add_argument("--speaker-data-csv", type=Path,
                        default=cejc_dir / "Speaker_data.csv")
    parser.add_argument("--output-csv", type=Path,
                        default=output_dir / "calm3-22b_selected_conversation.csv")
    parser.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--max-utterances", type=int, default=None,
                        help="Limit to first N utterances (e.g. 30).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    conversation_id = conversation_id_from_directory(
        args.cejc_root, FIXED_RUN_CONFIG.conversation_dir
    )
    target_age = normalize_age_label(FIXED_RUN_CONFIG.target_age)
    target_gender = normalize_gender_label(FIXED_RUN_CONFIG.target_gender)
    if not target_age or not target_gender:
        raise ValueError("target_age and target_gender in FIXED_RUN_CONFIG are required.")

    run_selected_conversation_analysis(
        conversation_id=conversation_id,
        cejc_root=args.cejc_root,
        speaker_data_csv=args.speaker_data_csv,
        relation_csv=args.relation_csv,
        target_age=target_age,
        target_gender=target_gender,
        output_csv=args.output_csv,
        model_id=args.model_id,
        max_utterances=FIXED_RUN_CONFIG.max_utterances,
    )


if __name__ == "__main__":
    main()