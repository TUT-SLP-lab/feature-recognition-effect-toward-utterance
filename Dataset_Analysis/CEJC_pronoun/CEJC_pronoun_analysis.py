#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First-person pronoun analysis for the CEJC corpus with HTML and distribution graphs.

Reuses the CEJC morphological CSV traversal from SFP_counter.py.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Load SFP_counter dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
SFP_COUNTER_DIR = SCRIPT_DIR.parent / "CEJC_SFP_Analyze"
SFP_COUNTER_PATH = SFP_COUNTER_DIR / "SFP_counter.py"

spec = importlib.util.spec_from_file_location("cejc_sfp_counter", SFP_COUNTER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load SFPCounter from {SFP_COUNTER_PATH}")

_sfp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_sfp_module)
SFPCounter = _sfp_module.SFPCounter

# Constants adapted from SFP_counter
AGE_ORDER = [
    '0-4歳',  '5-9歳',  '10-14歳', '15-19歳', '20-24歳', '25-29歳',
    '30-34歳', '35-39歳', '40-44歳', '45-49歳', '50-54歳', '55-59歳',
    '60-64歳', '65-69歳', '70-74歳', '75-79歳', '80-84歳', '85-89歳',
    '90-94歳', '95-99歳',
]
RELATIVE_AGE_THRESHOLD = 5

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
    """Count first-person pronouns in CEJC morph CSV files with advanced visualization."""

    # Screened list: strictly keeping high-confidence real-world pronouns
    CANONICAL_PRONOUNS = {
        "私":   {"私", "わたし", "ワタシ", "わたくし", "あたくし"},
        "あたし": {"あたし"},
        "僕":   {"僕", "ぼく", "ボク"},
        "俺":   {"俺", "おれ", "オレ"},
        "うち":  {"うち", "ウチ"},
    }

    def __init__(self, cejc_root: str, encoding: str = "cp932"):
        super().__init__(cejc_root, encoding=encoding)
        self.relation_data = self._load_csv("Speaker_Conversation_Relation.csv")
        self.speaker_meta = self._build_speaker_meta()
        self.conversation_speakers = self._build_conversation_speakers()
        self.label_to_speaker_id = self._build_label_to_speaker_id()
        self.conversation_speaker_records = self._build_conversation_speaker_records()

    def _build_speaker_meta(self) -> Dict[str, Dict[str, Any]]:
        meta: Dict[str, Dict[str, Any]] = {}
        if self.speaker_data.empty:
            return meta
        for _, row in self.speaker_data.iterrows():
            speaker_id = str(row.get("話者ID", "") or "").strip()
            if not speaker_id:
                continue
            meta[speaker_id] = {
                "name": str(row.get("話者名", "") or "").strip(),
                "age": str(row.get("年齢", "") or "").strip(),
                "gender": str(row.get("性別", "") or "").strip(),
            }
        return meta

    def _build_conversation_speakers(self) -> Dict[str, List[str]]:
        speakers_by_conv: Dict[str, List[str]] = defaultdict(list)
        if self.relation_data.empty:
            return speakers_by_conv
        seen: Dict[str, set] = defaultdict(set)
        for _, row in self.relation_data.iterrows():
            conv_id = str(row.get("会話ID", "") or "").strip()
            speaker_id = str(row.get("話者ID", "") or "").strip()
            if not conv_id or not speaker_id or speaker_id in seen[conv_id]:
                continue
            seen[conv_id].add(speaker_id)
            speakers_by_conv[conv_id].append(speaker_id)
        return speakers_by_conv

    def _build_label_to_speaker_id(self) -> Dict[str, Dict[str, str]]:
        mapping: Dict[str, Dict[str, str]] = defaultdict(dict)
        if self.relation_data.empty:
            return mapping
        for _, row in self.relation_data.iterrows():
            conv_id = str(row.get("会話ID", "") or "").strip()
            speaker_id = str(row.get("話者ID", "") or "").strip()
            speaker_label = str(row.get("話者ラベル", "") or "").strip()
            if not conv_id or not speaker_id:
                continue
            mapping[conv_id][speaker_id] = speaker_id
            if speaker_label:
                mapping[conv_id][speaker_label] = speaker_id
        return mapping

    def _build_conversation_speaker_records(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        records: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
        if self.relation_data.empty:
            return records
        for _, row in self.relation_data.iterrows():
            conv_id = str(row.get("会話ID", "") or "").strip()
            speaker_id = str(row.get("話者ID", "") or "").strip()
            speaker_label = str(row.get("話者ラベル", "") or "").strip()
            if not conv_id or not speaker_id:
                continue

            speaker_info = self.speaker_meta.get(speaker_id, {})
            record = {
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "speaker_name": speaker_info.get("name", "Unknown") or "Unknown",
                "speaker_age": speaker_info.get("age", "Unknown") or "Unknown",
                "speaker_gender": speaker_info.get("gender", "Unknown") or "Unknown",
            }
            records[conv_id][speaker_id] = record
            if speaker_label:
                records[conv_id][speaker_label] = record
        return records

    def _normalize_pronoun(self, surface: str, lemma: str, pos: str = "") -> Optional[str]:
        candidates = {str(surface or "").strip(), str(lemma or "").strip()}
        for canonical, variants in self.CANONICAL_PRONOUNS.items():
            if candidates & variants:
                # うち disambiguation: only accept if POS indicates pronoun
                if canonical == "うち" and "代名詞" not in pos:
                    return None
                return canonical
        return None

    def _resolve_speaker_id(self, conversation_id: str, speaker_label: str) -> str:
        speaker_label = str(speaker_label or "").strip()
        if not speaker_label:
            return "Unknown"
        conv_map = self.label_to_speaker_id.get(conversation_id, {})
        if speaker_label in conv_map:
            return conv_map[speaker_label]
        if speaker_label in self.speaker_meta:
            return speaker_label
        if "_" in speaker_label:
            base_id = speaker_label.split("_", 1)[0]
            if base_id in self.speaker_meta:
                return base_id
        return "Unknown"

    def _get_conversation_speaker_record(self, conversation_id: str, speaker_label: str) -> Dict[str, str]:
        conv_records = self.conversation_speaker_records.get(conversation_id, {})
        speaker_label = str(speaker_label or "").strip()
        if speaker_label in conv_records:
            return conv_records[speaker_label]
        speaker_id = self._resolve_speaker_id(conversation_id, speaker_label)
        if speaker_id in conv_records:
            return conv_records[speaker_id]
        if speaker_id != "Unknown":
            info = self.speaker_meta.get(speaker_id, {})
            return {
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "speaker_name": info.get("name", "Unknown"),
                "speaker_age": info.get("age", "Unknown"),
                "speaker_gender": info.get("gender", "Unknown"),
            }
        return {"speaker_id": "Unknown", "speaker_label": speaker_label, "speaker_name": "Unknown", "speaker_age": "Unknown", "speaker_gender": "Unknown"}

    def _conversation_opponent(self, conversation_id: str, speaker_id: str) -> Tuple[str, Dict[str, str]]:
        participants = self.conversation_speakers.get(conversation_id, [])
        if len(participants) != 2 or speaker_id not in participants:
            return "Unknown", {"name": "Unknown", "age": "Unknown", "gender": "Unknown"}
        opponent_id = participants[0] if participants[1] == speaker_id else participants[1]
        info = self.speaker_meta.get(opponent_id, {})
        return opponent_id, {"name": info.get("name", "Unknown"), "age": info.get("age", "Unknown"), "gender": info.get("gender", "Unknown")}

    
    def _is_quoted_speech(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Syntactical check for quoted speech, looking both forward for trailing markers
        and backward for introductory speech verbs across utterance boundaries.
        """
        # --- FORWARD CHECK (Trailing quote markers) ---
        # Expanded to 15 tokens to catch longer quotes like "俺 にんにく植えてきたとかってってゆって"
        for i in range(idx + 1, min(len(df), idx + 15)):
            surface = str(df.iloc[i].get("書字形", ""))
            pos = str(df.iloc[i].get("品詞", ""))
            
            if surface in ["って", "て"] and "助詞" in pos:
                # Expanded internal lookahead slightly to accommodate stacked particles ('ってってゆって')
                for j in range(i + 1, min(len(df), i + 4)):
                    next_lemma = str(df.iloc[j].get("語彙素", ""))
                    next_surface = str(df.iloc[j].get("書字形", ""))
                    if next_lemma in ["言う", "いう", "話す", "聞く", "呼ぶ", "思う", "おもう", "考える"] or next_surface in ["ゆっ", "ゆて", "ゆー", "言っ"]:
                        return True
                # If it's a floating quote particle without a verb attached, it's still highly likely a quote
                return True

            if surface == "と" and "助詞" in pos:
                 for j in range(i + 1, min(len(df), i + 4)):
                    next_lemma = str(df.iloc[j].get("語彙素", ""))
                    if next_lemma in ["言う", "いう", "話す", "聞く", "呼ぶ", "思う", "おもう", "考える"]:
                        return True
            
            # --- THE FIX ---
            # Removed raw strings like "か" and "わ" because they appear inside quotes (e.g., "とか"). 
            # Now relying strictly on the POS tag "終助詞" (Sentence Final Particle) or hard punctuation.
            if "終助詞" in pos or surface in ["。", "、", "？", "！"]:
                break

        # --- BACKWARD CHECK (Introductory quote markers) ---
        for i in range(idx - 1, max(-1, idx - 15), -1):
            prev_lemma = str(df.iloc[i].get("語彙素", ""))
            prev_surface = str(df.iloc[i].get("書字形", ""))
            
            if prev_lemma in ["言う", "いう", "話す"] or prev_surface in ["ゆっ", "言っ", "いっ", "ゆて", "ゆー"]:
                return True

        return False

    def count_pronouns(self, file_pattern: str = "**/*-morphSUW.csv", two_speaker_only: bool = True, verbose: bool = True) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], int]]:
        rows: List[Dict[str, Any]] = []
        word_counts: Dict[Tuple[str, str], int] = {}
        
        morph_files = sorted(self.cejc_root.glob(file_pattern))
        if two_speaker_only:
            morph_files = self._filter_two_speaker_files(morph_files, verbose)

        for i, filepath in enumerate(morph_files):
            conversation_id = filepath.stem.replace("-morphSUW", "")
            if verbose and (i + 1) % 50 == 0:
                print(f"Processing file {i + 1}/{len(morph_files)}: {filepath.name}")

            df = self._load_morph_file(filepath)
            if df.empty or "話者ラベル" not in df.columns:
                continue

            for spk_label, grp in df.groupby("話者ラベル"):
                word_counts[(str(spk_label), conversation_id)] = len(grp)

            for idx, row in df.iterrows():
                pronoun = self._normalize_pronoun(row.get("書字形", ""), row.get("語彙素", ""), str(row.get("品詞", "") or ""))
                if not pronoun:
                    continue

                speaker_label = str(row.get("話者ラベル", "") or "").strip()
                speaker_record = self._get_conversation_speaker_record(conversation_id, speaker_label)
                speaker_id = speaker_record.get("speaker_id", "Unknown")

                # --- NEW STRICT FILTER ---
                # In CEJC, designated main subjects have IDs like "IC01".
                # Partners have IDs with underscores like "IC01_01".
                # If there is an underscore, or it starts with 'z', skip it completely.
                if "_" in speaker_id or speaker_id.lower().startswith("z"):
                    continue
                # -------------------------

                # --- ANTI-QUOTATION FILTER ---
                # Throw out the pronoun if it looks like they are quoting someone else
                if self._is_quoted_speech(df, idx):
                    continue
                # -----------------------------

                opponent_id, opponent_info = self._conversation_opponent(conversation_id, speaker_id)
                if opponent_id.lower().startswith("z") or opponent_info["age"] == "Unknown":
                    continue

                # --- 2ND-PERSON TRAP FILTER ---
                # If a female speaker uses "僕" or "俺" when talking to a young boy, 
                # she is almost certainly addressing him ("You"), not herself.
                if speaker_record.get("speaker_gender") == "女性" and pronoun in ["僕", "俺"]:
                    if opponent_info["age"] in ["0-4歳", "5-9歳", "10-14歳"] and opponent_info["gender"] == "男性":
                        continue
                # ------------------------------

                rows.append({
                    "pronoun": pronoun,
                    "surface": str(row.get("書字形", "") or ""),
                    "lemma": str(row.get("語彙素", "") or ""),
                    "speaker": speaker_label,
                    "main_speaker_id": speaker_id,
                    "main_speaker_name": speaker_record.get("speaker_name"),
                    "main_gender": speaker_record.get("speaker_gender"),
                    "main_age": speaker_record.get("speaker_age"),
                    "opponent_id": opponent_id,
                    "partner_gender": opponent_info["gender"],
                    "partner_age": opponent_info["age"],
                    "conversation_id": conversation_id,
                    "context": "".join(self._get_preceding_context(df, idx, n=4)),
                })

                # Debugging Trap
                if speaker_record.get("speaker_gender") == "女性" and pronoun == "俺":
                    print(f"DEBUG: Female used '俺' in {conversation_id}")
                    print(f"Context: {''.join(self._get_preceding_context(df, idx, n=4))}【{row.get('書字形', '')}】")

        return rows, word_counts

    def analyze_pronoun_by_partner(self, df_results: pd.DataFrame) -> pd.DataFrame:
        if df_results.empty:
            return pd.DataFrame()
        df = df_results.copy()
        df['partner_age_category'] = df['partner_age'].apply(self._get_age_category)
        return df

    def _render_norm_table(self, df: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], row_col: str, pronouns: List[str], row_order: Optional[List[str]] = None) -> str:
        if df.empty:
            return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
        try:
            norm = self._normalize_by_word_count(df, word_counts, row_col, 'pronoun', speaker_col='speaker')
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

        body = f'<h1>Pronoun Analysis by Main Speaker — 話者別一人称分析</h1><p class="page-note">{len(sorted_speakers)} 名の主話者 · 値 = 1,000 語あたり使用数</p>{speaker_content_html}'
        out_file = output_dir / f'{prefix}_main_speaker_analysis.html'
        out_file.write_text(self._html_page('Pronoun Analysis by Main Speaker', '#3498db', body), encoding='utf-8')
        print(f"HTML file successfully saved to: {out_file}")


def plot_pronoun_selection_ratio_by_gender(counter_instance: PronounCounter, df_results: pd.DataFrame, output_dir: Path):
    """Plots 100% stacked bar charts showing normalized pronoun choices, separated by gender."""
    if df_results.empty:
        return

    # 1. Get partner data and relative age
    partner_df = counter_instance.analyze_pronoun_by_partner(df_results)
    if partner_df.empty:
        return
    
    partner_df = counter_instance._add_relative_age_col(partner_df)
    
    # X-axis ordering
    ordered_categories = [
        '男性\n年下 (younger)', '男性\n同年代 (same age)', '男性\n年上 (older)',
        '女性\n年下 (younger)', '女性\n同年代 (same age)', '女性\n年上 (older)'
    ]

    # 2. Loop through both Male and Female speakers
    for target_gender in ['男性', '女性']:
        gender_df = partner_df[
            (partner_df['main_gender'] == target_gender) & 
            (partner_df['relative_age'] != 'Unknown')
        ].copy()

        if gender_df.empty:
            continue

        # 3. Create a combined Opponent Category for the X-axis
        gender_df['opponent_category'] = gender_df['partner_gender'] + '\n' + gender_df['relative_age']

        # 4. Count occurrences per SPEAKER per category (Preventing Data Dominance Bias)
        speaker_counts = gender_df.groupby(['main_speaker_name', 'opponent_category', 'pronoun']).size().unstack(fill_value=0)

        # Filter only for pronouns relevant to this gender
        pronoun_candidates = MALE_PRONOUNS if target_gender == '男性' else FEMALE_PRONOUNS
        available_pronouns = [p for p in pronoun_candidates if p in speaker_counts.columns]
        if not available_pronouns:
            continue
            
        speaker_counts = speaker_counts[available_pronouns]

        # Clean out rows where they didn't speak at all
        row_sums = speaker_counts.sum(axis=1)
        speaker_counts = speaker_counts[row_sums > 0]

        # 5. NORMALIZATION STEP: Calculate ratio per individual speaker first
        speaker_ratios = speaker_counts.div(speaker_counts.sum(axis=1), axis=0) * 100

        # 6. Average those normalized ratios across all speakers in the category
        speaker_ratios = speaker_ratios.reset_index()
        selection_ratio = speaker_ratios.groupby('opponent_category')[available_pronouns].mean()

        # 7. Reorder the X-axis
        valid_order = [c for c in ordered_categories if c in selection_ratio.index]
        if not valid_order:
            continue
        selection_ratio = selection_ratio.reindex(valid_order)

        # 8. Plot the 100% Stacked Bar Chart
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
        ax.set_title(f'{gender_en} Speakers: Normalized Pronoun Selection Ratio', fontsize=16, pad=15)
        ax.set_ylabel('Average Usage Percentage (%)', fontsize=12)
        ax.set_xlabel('Opponent Demographic', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title='Pronoun Choice', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add percentage text labels inside the bars (only if > 0%)
        for c in ax.containers:
            labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

        # Save it
        plt.tight_layout()
        filename_gender = "male" if target_gender == '男性' else "female"
        plot_path = output_dir / f'{filename_gender}_pronoun_selection_ratio.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved normalized {filename_gender} selection ratio graph -> {plot_path.name}")
        
def main():
    parser = argparse.ArgumentParser(description="Count and analyze first-person pronouns cleanly.")
    parser.add_argument("--cejc-root", type=str, default="/home/ryuu/Ryu/Corpus/CEJC")
    parser.add_argument("--output-dir", type=str, default="/home/ryuu/Ryu/Dataset_Analysis/CEJC_pronoun/output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        counter = PronounCounter(args.cejc_root)
        rows, word_counts = counter.count_pronouns(two_speaker_only=True, verbose=True)

        if rows:
            df_results = pd.DataFrame(rows)
            print(f"Loaded {len(df_results)} pronoun entries.")
            
            counter.export_results_tabbed(df_results, word_counts, out_dir)
            print("Generating pronoun adaptation graphs...")
            plot_pronoun_selection_ratio_by_gender(counter, df_results, out_dir)
            print("Done processing everything smoothly!")
        else:
            print("No valid pronoun matching found.")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()