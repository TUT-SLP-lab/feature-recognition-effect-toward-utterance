import pandas as pd
from pathlib import Path


CORPUS_ROOT = Path(__file__).parent.parent / "CEJC"
OUTPUT_DIR = Path(__file__).parent
ENCODING = "shift_jis"
TAG_COL = "タグ付き書字形"
SENT_FLAG_COL = "文頭フラグ"
SURFACE_COL = "書字形"
SPEAKER_COL = "話者ラベル"
START_COL = "発話単位の開始時刻"
END_COL = "発話単位の終了時刻"
FILE_COL = "会話ID"


def load_morphsuw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding=ENCODING, dtype=str)


def build_marked_utterance(tokens: list) -> str:
    """Join surface forms, appending (F) immediately after each filler token."""
    parts = []
    for t in tokens:
        surface = t[SURFACE_COL] if pd.notna(t[SURFACE_COL]) else ""
        tag = t[TAG_COL] if pd.notna(t[TAG_COL]) else ""
        parts.append(surface + ("(F)" if "(F " in tag else ""))
    return "".join(parts)


def extract_utterances_with_filler(df: pd.DataFrame, source_file: str) -> list[dict]:
    """
    Group tokens into utterances (B = start, I = continuation).
    Return only utterances containing at least one (F ...) filler tag.
    """
    utterances = []
    current_tokens = []

    for _, row in df.iterrows():
        if row[SENT_FLAG_COL] == "B":
            if current_tokens:
                utterances.append(current_tokens)
            current_tokens = [row]
        else:
            current_tokens.append(row)

    if current_tokens:
        utterances.append(current_tokens)

    results = []
    for tokens in utterances:
        tags = [t[TAG_COL] if pd.notna(t[TAG_COL]) else "" for t in tokens]
        has_filler = any("(F " in tag for tag in tags)
        if not has_filler:
            continue

        filler_words = [
            t[SURFACE_COL]
            for t in tokens
            if pd.notna(t[TAG_COL]) and "(F " in t[TAG_COL]
        ]

        results.append({
            "source_file": source_file,
            "speaker": tokens[0][SPEAKER_COL],
            "start_time": tokens[0][START_COL],
            "end_time": tokens[0][END_COL],
            "utterance": build_marked_utterance(tokens),
            "filler_words": "|".join(filler_words),
            "tagged_forms": "|".join(tag for tag in tags if "(F " in tag),
            "token_count": len(tokens),
        })

    return results


def merge_consecutive_filler_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge back-to-back filler utterances from the same speaker in the same file.
    'Back-to-back' means the next utterance starts exactly where the previous one ended (gap = 0).
    Returns a new DataFrame where such runs are collapsed into a single row.
    """
    if df.empty:
        return df

    rows = df.to_dict("records")
    merged = []
    group = [rows[0]]

    for row in rows[1:]:
        prev = group[-1]
        same_file = row["source_file"] == prev["source_file"]
        same_speaker = row["speaker"] == prev["speaker"]
        adjacent = float(row["start_time"]) == float(prev["end_time"])

        if same_file and same_speaker and adjacent:
            group.append(row)
        else:
            merged.append(group)
            group = [row]
    merged.append(group)

    result = []
    for group in merged:
        if len(group) == 1:
            result.append(group[0])
        else:
            result.append({
                "source_file": group[0]["source_file"],
                "speaker": group[0]["speaker"],
                "start_time": group[0]["start_time"],
                "end_time": group[-1]["end_time"],
                "utterance": " / ".join(r["utterance"] for r in group),
                "filler_words": "|".join(
                    w for r in group for w in r["filler_words"].split("|")
                ),
                "tagged_forms": "|".join(
                    t for r in group for t in r["tagged_forms"].split("|")
                ),
                "token_count": sum(r["token_count"] for r in group),
                "merged_utterance_count": len(group),
            })

    out = pd.DataFrame(result)
    if "merged_utterance_count" not in out.columns:
        out["merged_utterance_count"] = 1
    out["merged_utterance_count"] = out["merged_utterance_count"].fillna(1).astype(int)
    return out


def main():
    all_results = []
    csv_files = sorted(CORPUS_ROOT.rglob("*-morphSUW.csv"))
    print(f"Found {len(csv_files)} morphSUW files")

    for path in csv_files:
        try:
            df = load_morphsuw(path)
            source = path.stem.replace("-morphSUW", "")
            results = extract_utterances_with_filler(df, source)
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")

    output_df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "utterances_with_filler.csv"
    output_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nExtracted {len(output_df)} utterances containing filler (F tag)")
    print(f"Saved to: {out_path}")

    merged_df = merge_consecutive_filler_utterances(output_df)
    merged_only = merged_df[merged_df["merged_utterance_count"] > 1].copy()
    merged_path = OUTPUT_DIR / "utterances_with_filler_merged.csv"
    merged_only.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"\nMerged {len(merged_only)} groups of back-to-back filler utterances")
    print(f"Saved to: {merged_path}")

    if not output_df.empty:
        print("\nFiller word frequency:")
        filler_series = output_df["filler_words"].str.split("|").explode()
        print(filler_series.value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
