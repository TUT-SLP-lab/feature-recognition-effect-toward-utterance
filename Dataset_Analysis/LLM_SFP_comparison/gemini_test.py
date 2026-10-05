import os
import csv
import re
import argparse
from datetime import datetime
from google import genai

client = None

# CEJC annotation tags and patterns
PAUSE_ANNOTATIONS = re.compile(r"^\d+(?:\.\d+)?$|^\.$")
MEANINGFUL_TEXT = re.compile(r"[一-龯ぁ-んァ-ヶA-Za-z0-9]")
ANNOTATION_RESIDUE = re.compile(r"[()<>@#%+]|\b[A-Z]\b")
AIZUCHI_SURFACES = frozenset({"うん", "ああ", "あー", "ね", "え", "◇", "あ", "うーん", "うんうん"})


def _find_matching_paren(text: str, start_index: int):
    """Return the matching ')' index for text[start_index] == '(' or None."""
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _clean_parenthetical_content(content: str) -> str:
    """Normalize a single CEJC parenthetical annotation block."""
    content = content.strip()
    if not content:
        return ""

    if PAUSE_ANNOTATIONS.fullmatch(content):
        return ""

    # Keep the readable side when the annotation has alternatives like A|B.
    if "|" in content:
        content = content.split("|")[-1]

    # Remove the annotation tag code itself, e.g. T, W, D, K, R.
    content = re.sub(r"^[A-Z]\s*", "", content)

    # Remove tag placeholders that should never reach the LLM.
    content = content.replace("＃", "").replace("#", "")
    content = content.replace("%", "").replace("＋", "").replace("+", "")
    content = content.replace(":", "").replace("：", "")

    # Recursively strip nested parenthetical annotations such as (T (X ...)).
    content = _strip_cejc_markup(content)
    return content.strip()


def _strip_cejc_markup(text: str) -> str:
    """Strip CEJC markup while keeping readable lexical content."""
    text = re.sub(r"@.*$", "", text)
    text = re.sub(r"<[^>]*>", "", text)

    result = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            end_index = _find_matching_paren(text, index)
            if end_index is None:
                index += 1
                continue

            inner_text = text[index + 1:end_index]
            cleaned_inner = _clean_parenthetical_content(inner_text)
            if cleaned_inner:
                result.append(cleaned_inner)
            index = end_index + 1
            continue

        if char == ")":
            index += 1
            continue

        result.append(char)
        index += 1

    cleaned = "".join(result)
    cleaned = cleaned.replace("%", "").replace("+", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def clean_for_llm(text: str) -> str:
    """Strip CEJC annotation tags and pause annotations."""
    return _strip_cejc_markup(text)

def should_passthrough(text: str) -> bool:
    """Skip utterance if it becomes empty or a backchannel after cleaning."""
    content_only = clean_for_llm(text)
    content_only = content_only.strip().rstrip("。、！？：:?.")

    if not content_only:
        return True
    if ANNOTATION_RESIDUE.search(content_only):
        return True
    if not MEANINGFUL_TEXT.search(content_only):
        return True
    if len(content_only) <= 3 and content_only in AIZUCHI_SURFACES:
        return True
    return False

def load_and_filter_utterances(luu_file_path):
    """Load utterances, filter (L) and backchannels"""
    utterances = []
    with open(luu_file_path, 'r', encoding='shift-jis') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text'].strip()
            if not text:
                continue
            # Include passthrough/backchannel utterances in the output
            is_passthrough = should_passthrough(text)
            cleaned_text = clean_for_llm(text)
            utterances.append({
                'luuID': row['luuID'],
                'speakerID': row['speakerID'],
                'original': text,
                'cleaned': cleaned_text,
                'passthrough': is_passthrough
            })
    return utterances

def transform_style(
    utterance_text,
    target_gender_age="40s male",
    opponent_gender_age="40s male",
    target_personality="polite",
):
    """Transform utterance to target style"""
    global client
    if client is None:
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("GENAI_API_KEY is not set")
        client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=f"change style of this sentence as if said by {target_gender_age} toward {opponent_gender_age} with {target_personality} personality: '{utterance_text}' output only the transformed sentence."
    )
    return response.text.strip()

CEJC_ROOT = "/home/ryuu/Ryu/Dataset_Analysis/CEJC"
OUT_CSV = "conversion_results.csv"
DEFAULT_LUU_FILE = "/home/ryuu/Ryu/Dataset_Analysis/CEJC/C001/C001_002/C001_002-luu.csv"
TARGET_STYLE_BY_SPEAKER_SUFFIX = {
    "01": "20s male",
    "02": "50s female",
}

TARGET_PERSONALITY_BY_SPEAKER_SUFFIX = {
    "01": "polite",
    "02": "friendly",
}


def get_target_style_for_speaker(speaker_id):
    """Resolve the target style for a speaker from the configured suffix map."""
    match = re.search(r"(?<!\d)(\d{2})(?!\d)", speaker_id)
    if match:
        code = match.group(1)
        if code in TARGET_STYLE_BY_SPEAKER_SUFFIX:
            return TARGET_STYLE_BY_SPEAKER_SUFFIX[code]
    return "40s male"


def get_target_personality_for_speaker(speaker_id):
    """Resolve the target personality for a speaker from the configured suffix map."""
    match = re.search(r"(?<!\d)(\d{2})(?!\d)", speaker_id)
    if match:
        code = match.group(1)
        if code in TARGET_PERSONALITY_BY_SPEAKER_SUFFIX:
            return TARGET_PERSONALITY_BY_SPEAKER_SUFFIX[code]
    return "polite"


def find_luu_files(root_dir):
    """Yield paths to files ending with '-luu.csv' under root_dir."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith("-luu.csv"):
                yield os.path.join(dirpath, fn)


def process_all_files(cejc_root, out_csv_path):
    """Process each luu file that has exactly 2 speakers and append results to CSV."""
    process_luu_paths(find_luu_files(cejc_root), out_csv_path)


def process_luu_paths(luu_paths, out_csv_path):
    """Process the given luu file paths and append results to CSV."""
    fieldnames = [
        'timestamp', 'luu_file', 'luuID', 'speakerID', 'original', 'cleaned', 'passthrough', 'transformed'
    ]

    # Overwrite the file each run so old results never remain mixed in.
    with open(out_csv_path, 'w', encoding='utf-8-sig', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for luu_path in luu_paths:
            try:
                utterances = load_and_filter_utterances(luu_path)
            except Exception as e:
                print(f"Failed to read {luu_path}: {e}")
                continue

            speakers = {utt['speakerID'] for utt in utterances}
            if len(speakers) != 2:
                print(f"Skipping {luu_path}: {len(speakers)} speakers (need 2)")
                continue

            speaker_styles = {speaker_id: get_target_style_for_speaker(speaker_id) for speaker_id in speakers}
            speaker_personalities = {
                speaker_id: get_target_personality_for_speaker(speaker_id)
                for speaker_id in speakers
            }

            print(f"Processing {luu_path} with 2 speakers, {len(utterances)} utterances")

            for utt in utterances:
                transformed = ''
                if utt.get('passthrough'):
                    transformed = '[PASSTHROUGH - not transformed]'
                else:
                    cleaned_text = utt['cleaned'].strip()
                    if not MEANINGFUL_TEXT.search(cleaned_text):
                        transformed = '[PASSTHROUGH - no meaningful content after cleaning]'
                    else:
                        # Gemini only sees cleaned content that passed the safety checks.
                        try:
                            opponent_gender_age = next(
                                style for speaker_id, style in speaker_styles.items()
                                if speaker_id != utt['speakerID']
                            )
                            target_style = get_target_style_for_speaker(utt['speakerID'])
                            target_personality = get_target_personality_for_speaker(utt['speakerID'])
                            transformed = transform_style(
                                cleaned_text,
                                target_style,
                                opponent_gender_age,
                                target_personality,
                            )
                        except Exception as e:
                            transformed = f'[ERROR: {e}]'

                row = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'luu_file': luu_path,
                    'luuID': utt.get('luuID'),
                    'speakerID': utt.get('speakerID'),
                    'original': utt.get('original'),
                    'cleaned': utt.get('cleaned'),
                    'passthrough': str(bool(utt.get('passthrough'))),
                    'transformed': transformed,
                }
                writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CEJC style conversion")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Process all CEJC luu files with exactly 2 speakers",
    )
    parser.add_argument(
        "--luu-file",
        default=DEFAULT_LUU_FILE,
        help="Single luu csv file to process when --run-all is not set",
    )
    parser.add_argument(
        "--out-csv",
        default=OUT_CSV,
        help="Output CSV path",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.run_all:
        process_all_files(CEJC_ROOT, args.out_csv)
        print(f"Done. Results written to {args.out_csv}")
    else:
        process_luu_paths([args.luu_file], args.out_csv)
        print(f"Done. Results written to {args.out_csv}")