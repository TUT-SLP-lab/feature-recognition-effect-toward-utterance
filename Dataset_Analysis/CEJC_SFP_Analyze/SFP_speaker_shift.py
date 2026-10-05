#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CEJC SFP Speaker Shift
一話者の会話ごとの終助詞ジェンダー強度の変化を可視化する

Case-study chart: for one main speaker who appears across multiple classified
conversations, show how their SFP gender-intensity tier composition (strongly/
moderately feminine, neutral, moderately/strongly masculine, none) shifts from
one conversation partner to the next.

Reuses the same sfp_ai_classification_*.csv outputs and the same 6-tier lexicon
format as SFP_counter_AI.py's --gender-intensity mode, but is a standalone script
since it answers a different question (one speaker's variation across partners)
rather than the corpus-wide age x gender summary.

Usage:
    python3 SFP_speaker_shift.py --main-speaker-id T008 \
        --run-dir output/ai_pilot/20260904_164225

Output:
    <run-dir>/overall/sfp_gender_intensity_by_conversation_<speaker>_with_ka.svg
    <run-dir>/overall/sfp_gender_intensity_by_conversation_<speaker>_no_ka.svg
    <run-dir>/overall/sfp_gender_intensity_by_conversation_<speaker>_with_ka.csv
    <run-dir>/overall/sfp_gender_intensity_by_conversation_<speaker>_no_ka.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

AI_OUTPUT_DIR = '/home/ryuu/Ryu/Dataset_Analysis/CEJC_SFP_Analyze/output/ai_pilot'

GENDER_INTENSITY_LEXICON_WITH_KA = (
    '/home/ryuu/Ryu/System_implement/LLM_Response/'
    'SFP_Bucket_by_gender_detailed.json'
)

GENDER_INTENSITY_TIERS = [
    'strongly_feminine', 'moderately_feminine', 'neutral',
    'moderately_masculine', 'strongly_masculine', 'none',
]
GENDER_INTENSITY_LABELS = {
    'strongly_feminine': 'Strongly feminine',
    'moderately_feminine': 'Moderately feminine',
    'neutral': 'Neutral',
    'moderately_masculine': 'Moderately masculine',
    'strongly_masculine': 'Strongly masculine',
    'none': 'None (unmatched)',
}
GENDER_INTENSITY_COLORS = {
    'strongly_feminine': '#a3272e',
    'moderately_feminine': '#e0918f',
    'neutral': '#d9d7cd',
    'moderately_masculine': '#86b6ef',
    'strongly_masculine': '#184f95',
    'none': '#c3c2b7',
}


def _load_gender_intensity_lexicon(lexicon_path: str) -> Dict[str, str]:
    """Flatten a {tier: [entry, ...]} lexicon JSON (entries may themselves be
    comma('、')-joined compound patterns) into a {individual_token: tier} lookup."""
    with open(lexicon_path, encoding='utf-8') as f:
        buckets = json.load(f)
    token_to_tier: Dict[str, str] = {}
    for tier, entries in buckets.items():
        for entry in entries:
            for tok in entry.split('、'):
                token_to_tier[tok.strip()] = tier
    return token_to_tier


def _normalize_sfp_entry(s: str) -> str:
    return (s.replace('（', '(').replace('）', ')')
             .replace('-ee', 'ee').replace('.ee', 'ee').strip())


def _classify_gender_intensity(entry, token_to_tier: Dict[str, str]) -> str:
    """Map one ai_matched_entry value to a gender-intensity tier, falling back to
    token-level voting on comma-split compounds; unmatched entries land in 'none'."""
    if not isinstance(entry, str) or not entry:
        return 'none'
    entry_n = _normalize_sfp_entry(entry)
    normalized_lookup = {_normalize_sfp_entry(k): v for k, v in token_to_tier.items()}
    if entry_n in normalized_lookup:
        return normalized_lookup[entry_n]

    votes: Dict[str, int] = {}
    for tok in entry.split('、'):
        tok_n = _normalize_sfp_entry(tok.strip())
        if tok_n in normalized_lookup:
            tier = normalized_lookup[tok_n]
            votes[tier] = votes.get(tier, 0) + 1
    if votes:
        return max(votes.items(), key=lambda kv: kv[1])[0]
    return 'none'


def build_speaker_conversation_intensity(
    ai_output_dir: str,
    main_speaker_id: str,
    lexicon_path: str = GENDER_INTENSITY_LEXICON_WITH_KA,
    exclude_particles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Pool every sfp_ai_classification_*.csv found under ai_output_dir, filter to one
    main_speaker_id, and compute the 6-tier gender-intensity % separately for each of
    that speaker's conversations — one row per conversation_id, carrying the partner's
    name/gender/age so the shift chart can order and label conversations by partner."""
    csv_paths = sorted(
        p for p in Path(ai_output_dir).glob('**/sfp_ai_classification_*.csv')
        if p.parent.name != 'overall'
    )
    if not csv_paths:
        print(f"No AI classification CSVs found under {ai_output_dir}")
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['main_speaker_id'] == main_speaker_id]
    if df.empty:
        print(f"No rows found for main_speaker_id={main_speaker_id}")
        return pd.DataFrame()

    if exclude_particles:
        df = df[~df['marked_particle'].isin(exclude_particles)]

    token_to_tier = _load_gender_intensity_lexicon(lexicon_path)

    def tier_for_row(row) -> str:
        if row['ai_bucket'] == 'none':
            return 'none'
        return _classify_gender_intensity(row.get('ai_matched_entry', ''), token_to_tier)

    df = df.copy()
    df['gender_intensity_tier'] = df.apply(tier_for_row, axis=1)

    meta = df.groupby('conversation_id').agg(
        main_gender=('main_gender', 'first'),
        main_age=('main_age', 'first'),
        partner_name=('partner_name', 'first'),
        partner_gender=('partner_gender', 'first'),
        partner_age=('partner_age', 'first'),
        n=('gender_intensity_tier', 'size'),
    )
    pct_table = (
        df.groupby('conversation_id')['gender_intensity_tier']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0) * 100
    ).reindex(columns=GENDER_INTENSITY_TIERS, fill_value=0.0)

    result = meta.join(pct_table).reset_index()
    return result


def build_speaker_partner_intensity(
    ai_output_dir: str,
    main_speaker_id: str,
    lexicon_path: str = GENDER_INTENSITY_LEXICON_WITH_KA,
    exclude_particles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Same pooling/classification as build_speaker_conversation_intensity, but
    aggregated one level up: one row per partner_name, pooling every conversation
    that speaker had with them. Utterance-weighted (not conversation-averaged), so a
    partner met across 5 long conversations isn't diluted to the same weight as a
    partner met once briefly — this is the 'stable register per relationship' view,
    smoothing out the conversation-to-conversation noise the per-conversation chart
    still shows."""
    csv_paths = sorted(
        p for p in Path(ai_output_dir).glob('**/sfp_ai_classification_*.csv')
        if p.parent.name != 'overall'
    )
    if not csv_paths:
        print(f"No AI classification CSVs found under {ai_output_dir}")
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df['main_speaker_id'] == main_speaker_id]
    if df.empty:
        print(f"No rows found for main_speaker_id={main_speaker_id}")
        return pd.DataFrame()

    if exclude_particles:
        df = df[~df['marked_particle'].isin(exclude_particles)]

    token_to_tier = _load_gender_intensity_lexicon(lexicon_path)

    def tier_for_row(row) -> str:
        if row['ai_bucket'] == 'none':
            return 'none'
        return _classify_gender_intensity(row.get('ai_matched_entry', ''), token_to_tier)

    df = df.copy()
    df['gender_intensity_tier'] = df.apply(tier_for_row, axis=1)

    meta = df.groupby('partner_name').agg(
        main_gender=('main_gender', 'first'),
        main_age=('main_age', 'first'),
        partner_gender=('partner_gender', 'first'),
        partner_age=('partner_age', 'first'),
        n_conversations=('conversation_id', 'nunique'),
        n=('gender_intensity_tier', 'size'),
        first_seen=('conversation_id', 'min'),
    )
    pct_table = (
        df.groupby('partner_name')['gender_intensity_tier']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0) * 100
    ).reindex(columns=GENDER_INTENSITY_TIERS, fill_value=0.0)

    result = meta.join(pct_table).reset_index().sort_values('first_seen')
    return result.drop(columns='first_seen').reset_index(drop=True)


def plot_speaker_partner_intensity(
    speaker_summary: pd.DataFrame,
    main_speaker_id: str,
    output_dir: str,
    variant_suffix: str = '',
):
    """One stacked-bar SVG: one bar per distinct conversation partner (pooling all of
    that speaker's conversations with them), showing the 6-tier gender-intensity
    composition — the 'per-relationship register' view, complementing the noisier
    per-conversation chart."""
    if speaker_summary.empty:
        print(f"No per-partner data available to plot for {main_speaker_id}.")
        return

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    sub = speaker_summary.copy()
    sub['partner_label'] = (
        sub['partner_name'] + '\n(' + sub['partner_gender'] + ', '
        + sub['partner_age'].str.replace('歳', '') + ')'
    )

    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 1.1), 6))
    bottom = pd.Series(0.0, index=sub.index)
    for tier in GENDER_INTENSITY_TIERS:
        values = sub[tier]
        bars = ax.bar(
            sub.index, values, bottom=bottom,
            label=GENDER_INTENSITY_LABELS[tier], color=GENDER_INTENSITY_COLORS[tier])
        for bar, pct in zip(bars, values):
            if pct >= 5:
                text_color = '#0b0b0b' if tier in ('neutral', 'moderately_masculine', 'none') else 'white'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                         f'{pct:.1f}%', ha='center', va='center', fontsize=8, color=text_color)
        bottom += values

    ax.set_ylabel('Percentage of SFP-marked utterances (%)')
    ax.set_xlabel('Conversation partner')
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [f"{row.partner_label}\n({row.n_conversations} conv., N={int(row.n)})" for row in sub.itertuples()],
        rotation=0, fontsize=8)
    ax.set_ylim(0, 100)

    main_gender = sub['main_gender'].iloc[0]
    main_age = sub['main_age'].iloc[0].replace('歳', '')
    title_tag_map = {'_with_ka': ' (か included)', '_no_ka': ' (か excluded)'}
    title_tag = title_tag_map.get(
        variant_suffix,
        f' ({variant_suffix.strip("_").replace("_", " ")})' if variant_suffix else '')
    ax.set_title(
        f'SFP Gender-Intensity by Conversation Partner — Speaker {main_speaker_id} '
        f'({main_gender}, {main_age}){title_tag}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)

    plot_path = Path(output_dir) / f'sfp_gender_intensity_by_partner_{main_speaker_id}{variant_suffix}.svg'
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved per-partner shift chart: {plot_path.name}")


def plot_speaker_conversation_intensity(
    speaker_summary: pd.DataFrame,
    main_speaker_id: str,
    output_dir: str,
    variant_suffix: str = '',
):
    """One stacked-bar SVG: every conversation a single main speaker appears in, each
    bar showing that conversation's 6-tier gender-intensity composition, ordered by
    partner (grouping repeat conversations with the same partner together) so a shift
    in style from one partner to another reads left-to-right."""
    if speaker_summary.empty:
        print(f"No per-conversation data available to plot for {main_speaker_id}.")
        return

    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    sub = speaker_summary.copy()
    sub['partner_label'] = (
        sub['partner_name'] + ' (' + sub['partner_gender'] + ', '
        + sub['partner_age'].str.replace('歳', '') + ')'
    )
    # Group same-partner conversations together, partners ordered by first appearance.
    partner_order = sub.drop_duplicates('partner_name')['partner_name'].tolist()
    sub['partner_rank'] = sub['partner_name'].map({p: i for i, p in enumerate(partner_order)})
    sub = sub.sort_values(['partner_rank', 'conversation_id']).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.6), 6))
    bottom = pd.Series(0.0, index=sub.index)
    for tier in GENDER_INTENSITY_TIERS:
        values = sub[tier]
        bars = ax.bar(
            sub.index, values, bottom=bottom,
            label=GENDER_INTENSITY_LABELS[tier], color=GENDER_INTENSITY_COLORS[tier])
        for bar, pct in zip(bars, values):
            if pct >= 6:
                text_color = '#0b0b0b' if tier in ('neutral', 'moderately_masculine', 'none') else 'white'
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                         f'{pct:.1f}%', ha='center', va='center', fontsize=7.5, color=text_color)
        bottom += values

    # Vertical separators between different partners.
    prev_partner = None
    for i, partner in enumerate(sub['partner_name']):
        if prev_partner is not None and partner != prev_partner:
            ax.axvline(i - 0.5, color='#898781', linewidth=0.8, linestyle='--', zorder=3)
        prev_partner = partner

    ax.set_ylabel('Percentage of SFP-marked utterances (%)')
    ax.set_xlabel('Conversation (grouped by partner)')
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [f"{row.conversation_id}\n{row.partner_label}\n(N={int(row.n)})" for row in sub.itertuples()],
        rotation=45, ha='right', fontsize=7)
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_ylim(0, 100)

    main_gender = sub['main_gender'].iloc[0]
    main_age = sub['main_age'].iloc[0].replace('歳', '')
    title_tag_map = {'_with_ka': ' (か included)', '_no_ka': ' (か excluded)'}
    title_tag = title_tag_map.get(
        variant_suffix,
        f' ({variant_suffix.strip("_").replace("_", " ")})' if variant_suffix else '')
    ax.set_title(
        f'SFP Gender-Intensity Shift Across Conversations — Speaker {main_speaker_id} '
        f'({main_gender}, {main_age}){title_tag}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8)

    plot_path = Path(output_dir) / f'sfp_gender_intensity_by_conversation_{main_speaker_id}{variant_suffix}.svg'
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved per-conversation shift chart: {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot one CEJC main speaker\'s SFP gender-intensity shift across their conversations.')
    parser.add_argument('--main-speaker-id', required=True,
                         help='main_speaker_id to trace, e.g. T008 (must appear in the classified CSVs).')
    parser.add_argument('--run-dir', default=None,
                         help='Directory to pool sfp_ai_classification_*.csv from (default: %s).' % AI_OUTPUT_DIR)
    parser.add_argument('--lexicon-path', default=GENDER_INTENSITY_LEXICON_WITH_KA,
                         help='Path to the {tier: [entry, ...]} JSON lexicon (default: %s).' % GENDER_INTENSITY_LEXICON_WITH_KA)
    args = parser.parse_args()

    source_dir = args.run_dir if args.run_dir else AI_OUTPUT_DIR
    overall_dir = Path(source_dir) / 'overall'
    overall_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ('_with_ka', None),
        ('_no_ka', ['か']),
    ]
    for variant_suffix, exclude_particles in variants:
        tag = 'か included' if exclude_particles is None else 'か excluded'
        print(f"\n--- {tag} ---")

        conv_summary = build_speaker_conversation_intensity(
            source_dir, args.main_speaker_id, args.lexicon_path, exclude_particles=exclude_particles)
        if not conv_summary.empty:
            csv_path = overall_dir / f'sfp_gender_intensity_by_conversation_{args.main_speaker_id}{variant_suffix}.csv'
            conv_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Saved per-conversation tier summary to: {csv_path}")
            plot_speaker_conversation_intensity(conv_summary, args.main_speaker_id, str(overall_dir), variant_suffix)

        partner_summary = build_speaker_partner_intensity(
            source_dir, args.main_speaker_id, args.lexicon_path, exclude_particles=exclude_particles)
        if not partner_summary.empty:
            csv_path = overall_dir / f'sfp_gender_intensity_by_partner_{args.main_speaker_id}{variant_suffix}.csv'
            partner_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"Saved per-partner tier summary to: {csv_path}")
            plot_speaker_partner_intensity(partner_summary, args.main_speaker_id, str(overall_dir), variant_suffix)


if __name__ == '__main__':
    main()
