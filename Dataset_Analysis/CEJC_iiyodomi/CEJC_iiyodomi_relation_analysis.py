import re
from pathlib import Path

import pandas as pd

CORPUS_ROOT = Path(__file__).parent.parent / "CEJC"
OUTPUT_DIR = Path(__file__).parent
SJIS = "shift_jis"
UTF8_BOM = "utf-8-sig"

CONV_CSV = CORPUS_ROOT / "Conversation.csv"
SPEAKER_CSV = CORPUS_ROOT / "Speaker_data.csv"
SCR_CSV = CORPUS_ROOT / "Speaker_Conversation_Relation.csv"
FILLER_CSV = OUTPUT_DIR / "utterances_with_filler.csv"
UTT_CACHE = OUTPUT_DIR / "total_utterance_counts.csv"


def age_midpoint(age_str) -> float | None:
    if pd.isna(age_str):
        return None
    m = re.match(r"(\d+)-(\d+)歳", str(age_str))
    return (int(m.group(1)) + int(m.group(2))) / 2.0 if m else None


def build_utterance_count_table(corpus_root: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"  Using cached utterance counts: {cache_path.name}")
        return pd.read_csv(cache_path, encoding=UTF8_BOM)

    print("  Scanning morphSUW files for utterance counts...")
    records = []
    for path in sorted(corpus_root.rglob("*-morphSUW.csv")):
        df = pd.read_csv(path, encoding=SJIS, dtype=str,
                         usecols=["会話ID", "文頭フラグ", "話者ラベル"])
        b = df[df["文頭フラグ"] == "B"]
        counts = b.groupby(["会話ID", "話者ラベル"]).size().reset_index(name="total_utterances")
        records.append(counts)

    result = pd.concat(records, ignore_index=True)
    result.to_csv(cache_path, index=False, encoding=UTF8_BOM)
    return result


def build_speaker_pairs(scr: pd.DataFrame, spk: pd.DataFrame,
                        two_p_ids: set) -> pd.DataFrame:
    """Self-join SCR to get both speaker and partner demographics per conversation."""
    scr_main = scr[
        scr["会話ID"].isin(two_p_ids)
        & ~scr["話者ラベル"].str.startswith("Z")
        & ~scr["話者ラベル"].str.startswith("N")
    ].copy()

    # Only keep conversations where exactly 2 non-anonymous speakers appear
    counts = scr_main.groupby("会話ID")["話者ラベル"].count()
    clean_ids = counts[counts == 2].index
    scr_clean = scr_main[scr_main["会話ID"].isin(clean_ids)]

    pairs = scr_clean.merge(scr_clean, on="会話ID", suffixes=("", "_partner"))
    pairs = pairs[pairs["話者ラベル"] != pairs["話者ラベル_partner"]]

    spk_demo = spk[["話者ID", "年齢", "性別"]].copy()
    spk_demo["age_mid"] = spk_demo["年齢"].apply(age_midpoint)

    pairs = pairs.merge(spk_demo, on="話者ID", how="left")
    pairs = pairs.merge(
        spk_demo.rename(columns={
            "話者ID": "話者ID_partner",
            "年齢": "年齢_partner",
            "性別": "性別_partner",
            "age_mid": "age_mid_partner",
        }),
        on="話者ID_partner",
        how="left",
    )
    return pairs


def _age_direction(row) -> str | None:
    s, p = row["age_mid"], row["age_mid_partner"]
    if pd.isna(s) or pd.isna(p):
        return None
    if s > p:
        return "older"
    if s < p:
        return "younger"
    return "same"


def build_analysis_df(filler: pd.DataFrame, conv_2p: pd.DataFrame,
                      pairs: pd.DataFrame, utt_counts: pd.DataFrame,
                      scr: pd.DataFrame, spk: pd.DataFrame) -> pd.DataFrame:
    # Filler utterance count per (source_file, speaker)
    filler_counts = (
        filler.groupby(["source_file", "speaker"])
        .size()
        .reset_index(name="filler_utterances")
    )

    # Join conversation metadata (relationship, duration)
    df = filler_counts.merge(
        conv_2p, left_on="source_file", right_on="会話ID", how="inner"
    )

    # Join total utterance counts
    df = df.merge(
        utt_counts.rename(columns={"話者ラベル": "speaker"}),
        left_on=["source_file", "speaker"],
        right_on=["会話ID", "speaker"],
        how="left",
        suffixes=("", "_utt"),
    )

    # Compute metrics
    df["filler_rate"] = df["filler_utterances"] / df["total_utterances"]
    df["filler_per_min"] = df["filler_utterances"] / df["会話時間"].astype(float)

    # Drop rows where normalization is impossible
    df = df[df["filler_rate"].notna() & df["filler_rate"].between(0, 1)]

    # Map speaker label → 話者ID via SCR
    speaker_lookup = scr[["会話ID", "話者ラベル", "話者ID"]].rename(
        columns={"話者ラベル": "speaker"}
    )
    df = df.merge(
        speaker_lookup,
        left_on=["source_file", "speaker"],
        right_on=["会話ID", "speaker"],
        how="left",
        suffixes=("", "_scr"),
    )

    # Join speaker demographics
    spk_demo = spk[["話者ID", "年齢", "性別"]].copy()
    spk_demo["age_mid"] = spk_demo["年齢"].apply(age_midpoint)
    df = df.merge(spk_demo, on="話者ID", how="left")

    # Join partner demographics from pairs table
    pairs_slim = pairs[
        ["会話ID", "話者ラベル", "年齢_partner", "性別_partner", "age_mid_partner"]
    ].rename(columns={"話者ラベル": "speaker"})
    df = df.merge(
        pairs_slim,
        left_on=["source_file", "speaker"],
        right_on=["会話ID", "speaker"],
        how="left",
        suffixes=("", "_pairs"),
    )

    # Derived columns
    df["age_direction"] = df.apply(_age_direction, axis=1)
    gender_map = {"女性": "F", "男性": "M"}
    df["gender_pair"] = (
        df["性別"].map(gender_map) + "→" + df["性別_partner"].map(gender_map)
    )
    df["age_group"] = (
        (df["age_mid"] // 10 * 10)
        .dropna()
        .astype(int)
        .astype(str)
        .apply(lambda x: x + "s")
        .reindex(df.index)
    )

    return df


def summarize(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    clean = df[df[group_col].notna()].copy()
    summary = (
        clean.groupby(group_col)
        .agg(
            mean_filler_rate=("filler_rate", "mean"),
            std_filler_rate=("filler_rate", "std"),
            mean_filler_per_min=("filler_per_min", "mean"),
            n_speakers=("filler_rate", "count"),
        )
        .reset_index()
        .sort_values("mean_filler_rate", ascending=False)
    )
    print(f"\n=== Filler rate by {label} ===")
    print(summary.to_string(index=False))
    return summary


def main():
    print("Loading data...")
    conv = pd.read_csv(CONV_CSV, encoding=SJIS)
    spk = pd.read_csv(SPEAKER_CSV, encoding=SJIS)
    scr = pd.read_csv(SCR_CSV, encoding=SJIS)
    filler = pd.read_csv(FILLER_CSV, encoding=UTF8_BOM)

    utt_counts = build_utterance_count_table(CORPUS_ROOT, UTT_CACHE)

    conv_2p = conv[conv["話者数"] == 2][["会話ID", "話者間の関係性", "会話時間"]].copy()
    two_p_ids = set(conv_2p["会話ID"])
    print(f"2-person conversations: {len(conv_2p)}")

    pairs = build_speaker_pairs(scr, spk, two_p_ids)

    df = build_analysis_df(filler, conv_2p, pairs, utt_counts, scr, spk)
    print(f"Analysis rows (speaker-conversation pairs): {len(df)}")

    full_path = OUTPUT_DIR / "speaker_filler_rates.csv"
    df.to_csv(full_path, index=False, encoding=UTF8_BOM)
    print(f"Full per-speaker table saved: {full_path.name}")

    analyses = [
        ("話者間の関係性", "relationship",  "filler_by_relationship.csv"),
        ("性別",           "gender",         "filler_by_gender.csv"),
        ("gender_pair",    "gender pair",    "filler_by_gender_pair.csv"),
        ("age_direction",  "age direction",  "filler_by_age_direction.csv"),
        ("age_group",      "age group",      "filler_by_age_group.csv"),
    ]
    for group_col, label, fname in analyses:
        summary = summarize(df, group_col, label)
        out = OUTPUT_DIR / fname
        summary.to_csv(out, index=False, encoding=UTF8_BOM)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
