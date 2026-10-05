#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First-person pronoun analysis for the Meidai/NUCC corpus.

Produces CEJC-style HTML reports and adaptation graphs by merging
the CEJC pronoun extraction logic with NUCC/Meidai metadata handling.
"""

from __future__ import annotations

import argparse
import traceback
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Reuse the Meidai XML streaming and HTML/data utilities
from nucc_SFP_analysis import SFPCounter, Token

AGE_ORDER = [
    '0-4歳',  '5-9歳',  '10-14歳', '15-19歳', '20-24歳', '25-29歳',
    '30-34歳', '35-39歳', '40-44歳', '45-49歳', '50-54歳', '55-59歳',
    '60-64歳', '65-69歳', '70-74歳', '75-79歳', '80-84歳', '85-89歳',
    '90-94歳', '95-99歳',
]

MALE_PRONOUNS   = ['俺', '僕', '私', 'あたし']
FEMALE_PRONOUNS = ['私', 'あたし', 'うち']

PRONOUN_COLORS = {
    '俺':   '#3498db',
    '僕':   '#2ecc71',
    '私':   '#9b59b6',
    'あたし': '#e74c3c',
    'うち':  '#f39c12',
}


class PronounCounter(SFPCounter):
    """Count first-person pronouns applying CEJC logic to Meidai XML utterances."""

    CANONICAL_PRONOUNS = {
        "私":   {"私", "わたし", "ワタシ", "わたくし", "あたくし"},
        "あたし": {"あたし"},
        "僕":   {"僕", "ぼく", "ボク"},
        "俺":   {"俺", "おれ", "オレ"},
        "うち":  {"うち", "ウチ"},
    }

    def _normalize_pronoun(self, surface: str, lemma: str, pos: str = "") -> Optional[str]:
        candidates = {str(surface or "").strip(), str(lemma or "").strip()}
        for canonical, variants in self.CANONICAL_PRONOUNS.items():
            if candidates & variants:
                # うち is a homonym (pronoun vs noun "house/inside"); accept only pronoun POS
                if canonical == "うち" and "代名詞" not in pos:
                    return None
                return canonical
        return None

    def _is_quoted_speech(self, tokens: Sequence[Token], idx: int) -> bool:
        """
        Syntactical check for quoted speech, mapped to NUCC Token objects.
        """
        # --- FORWARD CHECK (Trailing quote markers) ---
        for i in range(idx + 1, min(len(tokens), idx + 15)):
            surface = tokens[i].surface
            pos = tokens[i].pos
            
            if surface in ["って", "て"] and "助詞" in pos:
                for j in range(i + 1, min(len(tokens), i + 4)):
                    nt = tokens[j]
                    if nt.lemma in ["言う", "いう", "話す", "聞く", "呼ぶ", "思う", "おもう", "考える"] or nt.surface in ["ゆっ", "ゆて", "ゆー", "言っ"]:
                        return True
                return True

            if surface == "と" and "助詞" in pos:
                 for j in range(i + 1, min(len(tokens), i + 4)):
                    nt = tokens[j]
                    if nt.lemma in ["言う", "いう", "話す", "聞く", "呼ぶ", "思う", "おもう", "考える"]:
                        return True
            
            if "終助詞" in pos or surface in ["。", "、", "？", "！"]:
                break

        # --- BACKWARD CHECK (Introductory quote markers) ---
        for i in range(idx - 1, max(-1, idx - 15), -1):
            prev_lemma = tokens[i].lemma
            prev_surface = tokens[i].surface
            
            if prev_lemma in ["言う", "いう", "話す"] or prev_surface in ["ゆっ", "言っ", "いっ", "ゆて", "ゆー"]:
                return True

        return False

    def count_pronouns(self, conversation_filter: Optional[Sequence[str]] = None, verbose: bool = True) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], int]]:
        rows: List[Dict[str, Any]] = []
        word_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        conversation_filter_set = set(conversation_filter or [])

        if verbose:
            print("Streaming utterances from Meidai XML for pronouns...")

        for index, (meidai_meta, utterance_meta, tokens) in enumerate(self._iter_utterances(), start=1):
            conversation_id = meidai_meta.get("name", "")
            speaker_id = utterance_meta.get("s", "")
            
            if not self._is_two_speaker_conversation(meidai_meta):
                continue
            if conversation_filter_set and conversation_id not in conversation_filter_set:
                continue

            # Track Word Count exactly like SFP logic
            content_token_count = sum(1 for t in tokens if self._is_content_token(t))
            word_counts[(speaker_id, conversation_id)] += content_token_count

            for idx, token in enumerate(tokens):
                pronoun = self._normalize_pronoun(token.surface, token.lemma, token.pos)
                if not pronoun:
                    continue

                # Anti-Quotation Filter
                if self._is_quoted_speech(tokens, idx):
                    continue

                speaker_info = self.speaker_info.get(speaker_id, {})
                gender = utterance_meta.get("sex", speaker_info.get("gender", "Unknown"))
                age = utterance_meta.get("age", speaker_info.get("age", "Unknown"))

                rows.append({
                    "pronoun": pronoun,
                    "surface": token.surface,
                    "lemma": token.lemma,
                    "speaker": speaker_id,
                    "gender": gender,
                    "age": age,
                    "conversation_id": conversation_id,
                    "conversation_speakers": meidai_meta.get("speakers", ""),
                    "context": "".join(self._preceding_context(tokens, idx, n=4)),
                    "example_sentence": self._example_sentence(tokens, idx),
                })
                
            if verbose and index % 5000 == 0:
                print(f"Processed {index:,} utterances...")

        return rows, dict(word_counts)

    def analyze_pronoun_by_partner(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Format metadata (age buckets, gender formatting) and resolve opponents."""
        if df_results.empty:
            return pd.DataFrame()

        def _format_age(age_str):
            if pd.isna(age_str) or str(age_str).strip() == "" or str(age_str).lower() == "unknown":
                return "Unknown"
            match = re.search(r'\d+', str(age_str))
            if not match:
                return "Unknown"
            age_val = int(match.group())
            lower_bound = (age_val // 5) * 5
            upper_bound = lower_bound + 4
            return f"{lower_bound}-{upper_bound}歳"

        def _format_gender(g_str):
            g = str(g_str).upper()
            if g in ['F', 'FEMALE', '女性']: return '女性'
            if g in ['M', 'MALE', '男性']: return '男性'
            return 'Unknown'

        # Pre-resolve conversational pairs
        conv_partners = {}
        for conv_id in df_results['conversation_id'].unique():
            speakers_str = df_results[df_results['conversation_id'] == conv_id]['conversation_speakers'].iloc[0]
            speakers = [s.strip() for s in str(speakers_str).split(',')]
            if len(speakers) == 2:
                conv_partners[conv_id] = speakers

        results = []
        for _, row in df_results.iterrows():
            conv_id = row['conversation_id']
            if conv_id not in conv_partners:
                continue
            
            main_spk = row['speaker']
            partner = next((s for s in conv_partners[conv_id] if s != main_spk), None)
            if not partner:
                continue

            p_info = self.speaker_info.get(partner, {})
            p_gender = _format_gender(p_info.get('gender'))
            p_age = _format_age(p_info.get('age'))
            main_gender_fmt = _format_gender(row['gender'])

            # --- 2ND-PERSON TRAP FILTER ---
            if main_gender_fmt == '女性' and row['pronoun'] in ['僕', '俺']:
                if p_age in ["0-4歳", "5-9歳", "10-14歳"] and p_gender == "男性":
                    continue
            
            info = {
                'main_speaker_id': main_spk,
                'main_speaker_name': main_spk, 
                'main_gender': main_gender_fmt,
                'main_age': _format_age(row['age']),
                'partner_speaker_id': partner,
                'partner_gender': p_gender,
                'partner_age': p_age,
            }
            results.append({**row.to_dict(), **info})

        return pd.DataFrame(results)

    def _render_norm_table(self, df: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], row_col: str, pronouns: List[str], row_order: Optional[List[str]] = None) -> str:
        if df.empty:
            return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
        try:
            norm = self._normalize_by_word_count(df, word_counts, row_col, 'pronoun', speaker_col='speaker', conv_col='conversation_id')
            cols = [p for p in pronouns if p in norm.columns]
            if not cols:
                return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
            if row_order:
                norm = norm.reindex([r for r in row_order if r in norm.index])
            return norm[cols].to_html(float_format='%.2f', na_rep='-')
        except Exception as e:
            return f"<p style='color:red;font-size:13px;'>Error: {e}</p>\n"

    def export_results_tabbed(self, df_results: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], output_dir: Path, prefix: str = "pronoun_count"):
        partner_data = self.analyze_pronoun_by_partner(df_results)
        if partner_data.empty:
            return

        grouped = {sid: grp for sid, grp in partner_data.groupby('main_speaker_id')}
        sorted_speakers = sorted(grouped.items(), key=lambda kv: kv[1].iloc[0].get('main_speaker_name', kv[0]))

        speaker_content_html = ''

        for sid, spk_df in sorted_speakers:
            first = spk_df.iloc[0]
            name = first.get('main_speaker_name', sid)
            gender = first.get('main_gender', '?')
            pronoun_list = FEMALE_PRONOUNS if gender == '女性' else MALE_PRONOUNS
            age = first.get('main_age', '?')
            n_conv = spk_df['conversation_id'].nunique()
            n_pronouns = len(spk_df)

            pairs = spk_df[['speaker', 'conversation_id']].drop_duplicates()
            n_words = sum(word_counts.get((spk, conv), 0) for spk, conv in zip(pairs['speaker'], pairs['conversation_id']))

            content = f'''<div class="summary-box">
                <div class="stat-item"><span class="stat">{n_conv}</span><span class="stat-label">会話数<br>(conversations)</span></div>
                <div class="stat-item"><span class="stat">{n_words:,}</span><span class="stat-label">総語数<br>(words spoken)</span></div>
                <div class="stat-item"><span class="stat">{n_pronouns:,}</span><span class="stat-label">一人称総数<br>(Pronouns used)</span></div>
                <div class="stat-item"><span class="stat">{gender}</span><span class="stat-label">性別<br>(gender)</span></div>
                <div class="stat-item"><span class="stat" style="font-size:18px;">{age}</span><span class="stat-label">年齢<br>(age)</span></div>
            </div>
            <div style="padding:8px 0 0;">
            <p class="norm-note">すべての値 = {name} が 1,000 語あたりに使用する 一人称 回数</p>'''

            content += "<h3>相手の性別別 一人称使用率</h3>\n"
            content += self._render_norm_table(spk_df, word_counts, 'partner_gender', pronoun_list)

            valid_age_df = spk_df[spk_df['partner_age'] != 'Unknown'].copy()
            age_rows = [a for a in AGE_ORDER if a in valid_age_df['partner_age'].unique()]
            content += "<h3>相手の年齢別 一人称使用率</h3>\n"
            content += self._render_norm_table(valid_age_df, word_counts, 'partner_age', pronoun_list, row_order=age_rows)

            rel_df = self._add_relative_age_col(valid_age_df)
            rel_df = rel_df[rel_df['relative_age'] != 'Unknown']
            content += "<h3>相手の相対年齢別 一人称使用率 (±5歳基準)</h3>\n"
            content += self._render_norm_table(rel_df, word_counts, 'relative_age', pronoun_list, row_order=['年下 (younger)', '同年代 (same age)', '年上 (older)'])

            content += "</div>\n"
            speaker_content_html += f'<div class="speaker-panel"><h2>{name} — 一人称使用パターン（相手属性別）</h2>{content}</div>'

        body = f'<h1>NUCC Pronoun Analysis by Main Speaker — 話者別一人称分析</h1><p class="page-note">{len(sorted_speakers)} 名の主話者 · 値 = 1,000 語あたり使用数</p>{speaker_content_html}'
        out_file = output_dir / f'{prefix}_main_speaker_analysis.html'
        out_file.write_text(self._html_page('NUCC Pronoun Analysis', '#3498db', body), encoding='utf-8')
        print(f"HTML file successfully saved to: {out_file}")


def plot_pronoun_selection_ratio_by_gender(counter_instance: PronounCounter, df_results: pd.DataFrame, output_dir: Path):
    """Plots 100% stacked bar charts showing normalized pronoun choices, separated by gender."""
    if df_results.empty:
        return

    partner_df = counter_instance.analyze_pronoun_by_partner(df_results)
    if partner_df.empty:
        return
    
    partner_df = counter_instance._add_relative_age_col(partner_df)
    ordered_categories = [
        '男性\n年下 (younger)', '男性\n同年代 (same age)', '男性\n年上 (older)',
        '女性\n年下 (younger)', '女性\n同年代 (same age)', '女性\n年上 (older)'
    ]

    for target_gender in ['男性', '女性']:
        gender_df = partner_df[
            (partner_df['main_gender'] == target_gender) & 
            (partner_df['relative_age'] != 'Unknown')
        ].copy()

        if gender_df.empty:
            continue

        gender_df['opponent_category'] = gender_df['partner_gender'] + '\n' + gender_df['relative_age']
        speaker_counts = gender_df.groupby(['main_speaker_name', 'opponent_category', 'pronoun']).size().unstack(fill_value=0)

        pronoun_candidates = MALE_PRONOUNS if target_gender == '男性' else FEMALE_PRONOUNS
        available_pronouns = [p for p in pronoun_candidates if p in speaker_counts.columns]
        if not available_pronouns:
            continue
            
        speaker_counts = speaker_counts[available_pronouns]
        row_sums = speaker_counts.sum(axis=1)
        speaker_counts = speaker_counts[row_sums > 0]

        speaker_ratios = speaker_counts.div(speaker_counts.sum(axis=1), axis=0) * 100
        speaker_ratios = speaker_ratios.reset_index()
        selection_ratio = speaker_ratios.groupby('opponent_category')[available_pronouns].mean()

        valid_order = [c for c in ordered_categories if c in selection_ratio.index]
        if not valid_order:
            continue
        selection_ratio = selection_ratio.reindex(valid_order)

        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Meiryo', 'sans-serif']
        
        plot_colors = [PRONOUN_COLORS[p] for p in selection_ratio.columns]

        fig, ax = plt.subplots(figsize=(10, 6))
        
        selection_ratio.plot(
            kind='bar', 
            stacked=True, 
            color=plot_colors, 
            edgecolor='white',
            ax=ax
        )

        gender_en = "Male" if target_gender == '男性' else "Female"
        ax.set_title(f'NUCC {gender_en} Speakers: Normalized Pronoun Selection Ratio', fontsize=16, pad=15)
        ax.set_ylabel('Average Usage Percentage (%)', fontsize=12)
        ax.set_xlabel('Opponent Demographic', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Pronoun Choice', bbox_to_anchor=(1.05, 1), loc='upper left')

        for c in ax.containers:
            labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

        plt.tight_layout()
        filename_gender = "male" if target_gender == '男性' else "female"
        plot_path = output_dir / f'nucc_{filename_gender}_pronoun_selection_ratio.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved normalized {filename_gender} selection ratio graph -> {plot_path.name}")
        

def main():
    parser = argparse.ArgumentParser(description="Count and analyze first-person pronouns cleanly for NUCC/Meidai corpus.")
    parser.add_argument("--meidai-root", type=str, default="/home/ryuu/Ryu/Corpus/nucc/Himawari_nucc/Himawari_meidai")
    parser.add_argument("--output-dir", type=str, default="/home/ryuu/Ryu/Dataset_Analysis/nucc_analysis/output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        counter = PronounCounter(args.meidai_root)
        rows, word_counts = counter.count_pronouns(verbose=True)

        if rows:
            df_results = pd.DataFrame(rows)
            print(f"Loaded {len(df_results)} pronoun entries.")
            
            counter.export_results_tabbed(df_results, word_counts, out_dir, prefix="nucc_pronoun")
            print("Generating pronoun adaptation graphs...")
            plot_pronoun_selection_ratio_by_gender(counter, df_results, out_dir)
            print("Done processing everything smoothly!")
        else:
            print("No valid pronoun matching found.")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()