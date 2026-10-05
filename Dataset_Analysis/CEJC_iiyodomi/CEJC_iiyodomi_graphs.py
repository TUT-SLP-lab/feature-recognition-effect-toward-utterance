#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iiyodomi (filler) frequency graphs for the CEJC corpus.
Reads speaker_filler_rates.csv produced by CEJC_iiyodomi_relation_analysis.py.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_FILE  = Path(__file__).parent / "speaker_filler_rates.csv"
UTF8_BOM   = "utf-8-sig"

FONT_SETUP = ["Noto Sans CJK JP", "IPAGothic", "Hiragino Maru Gothic Pro", "Meiryo", "sans-serif"]

RELATIONSHIP_ORDER = [
    "家族", "友人知人", "元同僚", "同僚", "仕事関係", "先生生徒", "サービス場面関係", "親戚",
]

AGE_GROUP_ORDER = ["10s", "20s", "30s", "40s", "50s", "60s", "70s", "80s"]

RELATIVE_AGE_ORDER = ["younger", "same", "older"]
RELATIVE_AGE_LABELS = {
    "younger": "年下 (younger)",
    "same":    "同年代 (same)",
    "older":   "年上 (older)",
}

GENDER_PAIR_ORDER = ["F→F", "F→M", "M→F", "M→M"]
GENDER_PAIR_LABELS = {
    "F→F": "F → F",
    "F→M": "F → M",
    "M→F": "M → F",
    "M→M": "M → M",
}


def _setup_fonts():
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = FONT_SETUP


def _save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.name}")


# ---------------------------------------------------------------------------
# 1. Boxplot + strip: filler rate by relationship type
# ---------------------------------------------------------------------------
def plot_by_relationship(df: pd.DataFrame):
    col = "話者間の関係性"
    data = df[df[col].notna()].copy()

    present = [r for r in RELATIONSHIP_ORDER if r in data[col].unique()]
    other   = [r for r in data[col].unique() if r not in present]
    order   = present + other

    _setup_fonts()
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=data, x=col, y="filler_rate", order=order,
                palette="pastel", width=0.5, fliersize=0, ax=ax)
    sns.stripplot(data=data, x=col, y="filler_rate", order=order,
                  color=".3", alpha=0.4, size=4, jitter=True, ax=ax)

    ax.set_title("Filler Rate by Relationship Type\n話者間の関係性別 いよどみ率", fontsize=14, pad=12)
    ax.set_xlabel("話者間の関係性 (Relationship)", fontsize=11)
    ax.set_ylabel("Filler Rate (fillers / total utterances)", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    _save(fig, "filler_by_relationship.png")


# ---------------------------------------------------------------------------
# 2. Boxplot + strip: filler rate by speaker gender
# ---------------------------------------------------------------------------
def plot_by_gender(df: pd.DataFrame):
    col = "性別"
    data = df[df[col].isin(["女性", "男性"])].copy()

    _setup_fonts()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=data, x=col, y="filler_rate", order=["女性", "男性"],
                palette={"女性": "#e74c3c", "男性": "#3498db"},
                width=0.45, fliersize=0, ax=ax)
    sns.stripplot(data=data, x=col, y="filler_rate", order=["女性", "男性"],
                  color=".3", alpha=0.4, size=4, jitter=True, ax=ax)

    ax.set_title("Filler Rate by Speaker Gender\n話者性別別 いよどみ率", fontsize=14, pad=12)
    ax.set_xlabel("性別 (Gender)", fontsize=11)
    ax.set_ylabel("Filler Rate (fillers / total utterances)", fontsize=11)
    plt.tight_layout()
    _save(fig, "filler_by_gender.png")


# ---------------------------------------------------------------------------
# 3. Grouped boxplot: filler rate by gender dyad (F→F / F→M / M→F / M→M)
# ---------------------------------------------------------------------------
def plot_by_gender_pair(df: pd.DataFrame):
    col = "gender_pair"
    data = df[df[col].isin(GENDER_PAIR_ORDER)].copy()
    data["gender_pair_label"] = data[col].map(GENDER_PAIR_LABELS)
    order_labels = [GENDER_PAIR_LABELS[k] for k in GENDER_PAIR_ORDER]

    palette = {
        "F → F": "#e74c3c",
        "F → M": "#f39c12",
        "M → F": "#2ecc71",
        "M → M": "#3498db",
    }

    _setup_fonts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=data, x="gender_pair_label", y="filler_rate",
                order=order_labels, palette=palette, width=0.5, fliersize=0, ax=ax)
    sns.stripplot(data=data, x="gender_pair_label", y="filler_rate",
                  order=order_labels, color=".3", alpha=0.4, size=4, jitter=True, ax=ax)

    ax.set_title("Filler Rate by Gender Dyad\n性別ペア別 いよどみ率 (speaker → partner)", fontsize=14, pad=12)
    ax.set_xlabel("Gender Pair (speaker → partner)", fontsize=11)
    ax.set_ylabel("Filler Rate (fillers / total utterances)", fontsize=11)
    plt.tight_layout()
    _save(fig, "filler_by_gender_pair.png")


# ---------------------------------------------------------------------------
# 4. Boxplot + strip: filler rate by relative age direction (older/same/younger)
# ---------------------------------------------------------------------------
def plot_by_age_direction(df: pd.DataFrame):
    col = "age_direction"
    data = df[df[col].isin(RELATIVE_AGE_ORDER)].copy()
    data["age_dir_label"] = data[col].map(RELATIVE_AGE_LABELS)
    order_labels = [RELATIVE_AGE_LABELS[k] for k in RELATIVE_AGE_ORDER]

    _setup_fonts()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(data=data, x="age_dir_label", y="filler_rate",
                order=order_labels, palette="pastel", width=0.45, fliersize=0, ax=ax)
    sns.stripplot(data=data, x="age_dir_label", y="filler_rate",
                  order=order_labels, color=".3", alpha=0.4, size=4, jitter=True, ax=ax)

    ax.set_title("Filler Rate by Relative Age Direction\n相対年齢別 いよどみ率 (speaker vs partner)", fontsize=14, pad=12)
    ax.set_xlabel("Speaker Age Relative to Partner", fontsize=11)
    ax.set_ylabel("Filler Rate (fillers / total utterances)", fontsize=11)
    plt.tight_layout()
    _save(fig, "filler_by_age_direction.png")


# ---------------------------------------------------------------------------
# 5. Boxplot: filler rate by speaker age group (decade)
# ---------------------------------------------------------------------------
def plot_by_age_group(df: pd.DataFrame):
    col = "age_group"
    data = df[df[col].notna()].copy()
    present_order = [g for g in AGE_GROUP_ORDER if g in data[col].unique()]

    _setup_fonts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x=col, y="filler_rate", order=present_order,
                palette="Blues", width=0.5, fliersize=0, ax=ax)
    sns.stripplot(data=data, x=col, y="filler_rate", order=present_order,
                  color=".3", alpha=0.4, size=4, jitter=True, ax=ax)

    ax.set_title("Filler Rate by Speaker Age Group\n話者年代別 いよどみ率", fontsize=14, pad=12)
    ax.set_xlabel("Speaker Age Group", fontsize=11)
    ax.set_ylabel("Filler Rate (fillers / total utterances)", fontsize=11)
    plt.tight_layout()
    _save(fig, "filler_by_age_group.png")


# ---------------------------------------------------------------------------
# 6. FacetGrid: filler rate by relationship × speaker gender (side-by-side)
# ---------------------------------------------------------------------------
def plot_relationship_by_gender(df: pd.DataFrame):
    rel_col = "話者間の関係性"
    gen_col = "性別"
    data = df[df[rel_col].notna() & df[gen_col].isin(["女性", "男性"])].copy()

    present = [r for r in RELATIONSHIP_ORDER if r in data[rel_col].unique()]
    other   = [r for r in data[rel_col].unique() if r not in present]
    order   = present + other

    _setup_fonts()
    g = sns.catplot(
        data=data,
        x=rel_col, y="filler_rate",
        col=gen_col, col_order=["女性", "男性"],
        kind="box",
        order=order,
        palette="pastel",
        height=5, aspect=1.4,
        fliersize=0,
    )
    g.map_dataframe(
        sns.stripplot,
        x=rel_col, y="filler_rate",
        order=order,
        color=".3", alpha=0.35, size=3.5, jitter=True,
    )
    g.set_axis_labels("話者間の関係性 (Relationship)", "Filler Rate")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=30, ha="right")
    g.figure.suptitle(
        "Filler Rate by Relationship × Gender\n関係性 × 性別 いよどみ率",
        fontsize=14, y=1.03,
    )
    plt.tight_layout()
    _save(g.figure, "filler_relationship_by_gender.png")


# ---------------------------------------------------------------------------
# 7. FacetGrid: filler rate by relative age × speaker gender
# ---------------------------------------------------------------------------
def plot_age_direction_by_gender(df: pd.DataFrame):
    age_col = "age_direction"
    gen_col = "性別"
    data = df[df[age_col].isin(RELATIVE_AGE_ORDER) & df[gen_col].isin(["女性", "男性"])].copy()
    data["age_dir_label"] = data[age_col].map(RELATIVE_AGE_LABELS)
    order_labels = [RELATIVE_AGE_LABELS[k] for k in RELATIVE_AGE_ORDER]

    _setup_fonts()
    g = sns.catplot(
        data=data,
        x="age_dir_label", y="filler_rate",
        col=gen_col, col_order=["女性", "男性"],
        kind="box",
        order=order_labels,
        palette="pastel",
        height=5, aspect=1.1,
        fliersize=0,
    )
    g.map_dataframe(
        sns.stripplot,
        x="age_dir_label", y="filler_rate",
        order=order_labels,
        color=".3", alpha=0.4, size=4, jitter=True,
    )
    g.set_axis_labels("Relative Age (speaker vs partner)", "Filler Rate")
    g.set_titles("{col_name}")
    g.figure.suptitle(
        "Filler Rate by Relative Age × Gender\n相対年齢 × 性別 いよどみ率",
        fontsize=14, y=1.03,
    )
    plt.tight_layout()
    _save(g.figure, "filler_age_direction_by_gender.png")


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found.")
        print("Run CEJC_iiyodomi_relation_analysis.py first.")
        return

    df = pd.read_csv(DATA_FILE, encoding=UTF8_BOM)
    print(f"Loaded {len(df)} rows from {DATA_FILE.name}")

    plot_by_relationship(df)
    plot_by_gender(df)
    plot_by_gender_pair(df)
    plot_by_age_direction(df)
    plot_by_age_group(df)
    plot_relationship_by_gender(df)
    plot_age_direction_by_gender(df)

    print(f"\nAll graphs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
