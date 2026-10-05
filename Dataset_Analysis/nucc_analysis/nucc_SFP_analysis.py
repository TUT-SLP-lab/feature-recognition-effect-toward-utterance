#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentence-final particle counter for the Himawari Meidai corpus.

This version maintains Meidai/NUCC XML parsing but aligns the metadata and 
data outputs to seamlessly feed into the CEJC formatting, HTML generation, 
and Seaborn plotting pipelines.
"""

from __future__ import annotations

import argparse
import ast
import html
import re
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# ==============================================================
# CEJC REPORT CONSTANTS (Brought over for identical plotting)
# ==============================================================
REPORT_PARTICLES = [
    'ね', 'よ', 'か', 'よね', 'の', 'な', 'さ', 'じゃん', 'っけ', 'もん', 'わ', 'や', 'やん',
]

AGE_ORDER = [
    '0-4歳',  '5-9歳',  '10-14歳', '15-19歳', '20-24歳', '25-29歳',
    '30-34歳', '35-39歳', '40-44歳', '45-49歳', '50-54歳', '55-59歳',
    '60-64歳', '65-69歳', '70-74歳', '75-79歳', '80-84歳', '85-89歳',
    '90-94歳', '95-99歳',
]

RELATIVE_AGE_THRESHOLD   = 5   
CONTEXT_BEFORE           = 8   
CONTEXT_AFTER            = 2   


@dataclass(frozen=True)
class Token:
    """One token inside a Meidai utterance."""
    surface: str
    lemma: str
    pos: str
    reading: str = ""
    conj_form: str = ""
    conj_type: str = ""


class SFPCounter:
    """Sentence-final particle counter for the Meidai corpus."""

    TARGET_SFP_CANONICAL = {
        "よ", "ね", "よね", "か", "かい", "さ", "じゃん", "な",
        "の", "わ", "もの", "っけ", "かしら", "や", "やん", "ねん",
    }

    SFP_POS = "助詞-終助詞"
    ADVERBIAL_PARTICLE_POS = "助詞-副助詞"
    CASE_PARTICLE_POS = "助詞-格助詞"
    PUNCT_PREFIX = "補助記号"

    def __init__(self, meidai_root: str, encoding: str = "utf-16"):
        self.meidai_root = Path(meidai_root)
        self.encoding = encoding
        self.corpus_file = self._resolve_corpus_file(self.meidai_root)
        self.speaker_info: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _resolve_corpus_file(root: Path) -> Path:
        if root.is_file():
            return root
        candidates = [
            root / "Corpora" / "Meidai" / "corpus.xml",
            root / "corpus.xml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find Meidai corpus XML under {root}. Expected Corpora/Meidai/corpus.xml"
        )

    @staticmethod
    def _strip_outer_punctuation(text: str) -> str:
        return text.strip().strip("()（）[]【】「」『』《》〈〉<>・,，.。!！?？…〜~")

    def _normalize_particle(self, token: Token) -> str:
        surface = self._strip_outer_punctuation(token.surface)
        lemma = self._strip_outer_punctuation(token.lemma)
        if surface in self.TARGET_SFP_CANONICAL: return surface
        if lemma in self.TARGET_SFP_CANONICAL: return lemma
        if surface.endswith("ー") and surface[:-1] in self.TARGET_SFP_CANONICAL: return surface[:-1]
        if lemma.endswith("ー") and lemma[:-1] in self.TARGET_SFP_CANONICAL: return lemma[:-1]
        if surface == "ん" and lemma == "の": return "の"
        if surface == "もん" or lemma == "もの": return "もの"
        if surface == "け" and token.pos == self.SFP_POS: return "っけ"
        return surface or lemma

    def _canonical_target_set(self, target_sfps: Optional[Sequence[str]]) -> Optional[set[str]]:
        if target_sfps is None: return None
        return {self._normalize_target_particle(p) for p in target_sfps}

    def _normalize_target_particle(self, particle: str) -> str:
        particle = self._strip_outer_punctuation(particle)
        if particle in self.TARGET_SFP_CANONICAL: return particle
        if particle.endswith("ー") and particle[:-1] in self.TARGET_SFP_CANONICAL: return particle[:-1]
        if particle == "け": return "っけ"
        if particle == "もん": return "もの"
        if particle == "ん": return "の"
        return particle

    def _is_content_token(self, token: Token) -> bool:
        if not token.surface: return False
        if token.surface.startswith("himawari_") or token.lemma.startswith("himawari_"): return False
        return not token.pos.startswith(self.PUNCT_PREFIX)

    def _last_content_index(self, tokens: Sequence[Token]) -> int:
        for idx in range(len(tokens) - 1, -1, -1):
            if self._is_content_token(tokens[idx]):
                return idx
        return -1

    def _detect_compound_yone(self, tokens: Sequence[Token], idx: int) -> bool:
        if idx + 1 >= len(tokens): return False
        current, next_t = tokens[idx], tokens[idx + 1]
        return (current.surface == "よ" and current.pos == self.SFP_POS and 
                next_t.surface == "ね" and next_t.pos == self.SFP_POS)

    def _detect_toka_compound(self, tokens: Sequence[Token], idx: int) -> bool:
        if idx < 1: return False
        prev_t, current = tokens[idx - 1], tokens[idx]
        return (prev_t.surface == "と" and prev_t.pos == self.CASE_PARTICLE_POS and 
                current.surface == "か" and current.pos == self.ADVERBIAL_PARTICLE_POS)

    @staticmethod
    def _gender_from_speaker_id(speaker_id: str) -> str:
        """Infer gender from NUCC speaker ID: M### → 男性, F### → 女性."""
        if re.match(r'M\d+', speaker_id): return '男性'
        if re.match(r'F\d+', speaker_id): return '女性'
        return 'Unknown'

    @staticmethod
    def _is_two_speaker_conversation(meidai_meta: Dict[str, str]) -> bool:
        speakers = (meidai_meta.get("speakers") or "").strip()
        if not speakers: return False
        speaker_ids = [s.strip() for s in speakers.split(",") if s.strip()]
        return len(speaker_ids) == 2

    @staticmethod
    def _example_sentence(tokens: Sequence[Token], idx: int, n_before: int = 8, n_after: int = 2) -> str:
        start_idx = max(0, idx - n_before)
        end_idx = min(len(tokens), idx + n_after + 1)
        pieces = [token.surface for token in tokens[start_idx:end_idx]]
        if 0 <= idx - start_idx < len(pieces):
            pieces[idx - start_idx] = f"【{pieces[idx - start_idx]}】"
        return "".join(pieces)

    @staticmethod
    def _preceding_context(tokens: Sequence[Token], idx: int, n: int = 3) -> List[str]:
        start_idx = max(0, idx - n)
        return [token.surface for token in tokens[start_idx:idx]]

    def _iter_utterances(self) -> Iterator[Tuple[Dict[str, str], Dict[str, str], List[Token]]]:
        current_meidai: Dict[str, str] = {}
        for event, elem in ET.iterparse(self.corpus_file, events=("start", "end")):
            if event == "start" and elem.tag == "meidai":
                current_meidai = dict(elem.attrib)
            elif event == "end" and elem.tag == "u":
                utterance_meta = dict(elem.attrib)
                tokens: List[Token] = []
                for child in elem:
                    if child.tag != "s": continue
                    token = Token(
                        surface=(child.attrib.get("t") or child.attrib.get("s") or "").strip(),
                        lemma=(child.attrib.get("l") or "").strip(),
                        pos=(child.attrib.get("p") or "").strip(),
                        reading=(child.attrib.get("e") or "").strip(),
                        conj_form=(child.attrib.get("f") or "").strip(),
                        conj_type=(child.attrib.get("c") or "").strip(),
                    )
                    if token.surface.startswith("himawari_") or token.lemma.startswith("himawari_"): continue
                    tokens.append(token)

                if tokens:
                    speaker_id = utterance_meta.get("s", "")
                    if speaker_id and speaker_id not in self.speaker_info:
                        id_gender = self._gender_from_speaker_id(speaker_id)
                        attr_gender = utterance_meta.get("sex", "") or ""
                        self.speaker_info[speaker_id] = {
                            "speaker": speaker_id,
                            "gender": id_gender if id_gender != 'Unknown' else (attr_gender or 'Unknown'),
                            "age": utterance_meta.get("age", "Unknown") or "Unknown",
                        }
                    yield current_meidai.copy(), utterance_meta, tokens
                elem.clear()
            elif event == "end" and elem.tag == "meidai":
                elem.clear()

    def count_sfp_in_utterance(
        self, meidai_meta: Dict[str, str], utterance_meta: Dict[str, str], tokens: Sequence[Token],
        target_sfps: Optional[Sequence[str]] = None, include_compounds: bool = True, utterance_final_only: bool = True,
    ) -> Dict[str, Any]:
        if not tokens: return {"total": 0, "counts": {}, "details": []}

        counts = defaultdict(int)
        details = []
        processed_indices = set()
        canonical_targets = self._canonical_target_set(target_sfps)
        final_idx = self._last_content_index(tokens) if utterance_final_only else -1

        for idx, token in enumerate(tokens):
            if idx in processed_indices: continue

            normalized_particle = self._normalize_particle(token)

            if include_compounds and token.pos == self.SFP_POS and self._detect_compound_yone(tokens, idx):
                if utterance_final_only and idx + 1 != final_idx: continue
                particle = "よね"
                if canonical_targets is not None and particle not in canonical_targets: continue
                counts[particle] += 1
                details.append(self._build_detail_row(meidai_meta, utterance_meta, tokens, idx, particle, self.SFP_POS))
                processed_indices.update({idx, idx + 1})
                continue

            if include_compounds and token.pos == self.ADVERBIAL_PARTICLE_POS and self._detect_toka_compound(tokens, idx):
                if utterance_final_only and idx != final_idx: continue
                particle = "とか"
                if canonical_targets is not None and particle not in canonical_targets: continue
                counts[particle] += 1
                details.append(self._build_detail_row(meidai_meta, utterance_meta, tokens, idx, particle, self.ADVERBIAL_PARTICLE_POS))
                processed_indices.add(idx)
                continue

            if token.pos != self.SFP_POS: continue
            if utterance_final_only and idx != final_idx: continue
            if canonical_targets is not None and normalized_particle not in canonical_targets: continue

            counts[normalized_particle] += 1
            details.append(self._build_detail_row(meidai_meta, utterance_meta, tokens, idx, normalized_particle, token.pos))
            processed_indices.add(idx)

        return {"total": sum(counts.values()), "counts": dict(counts), "details": details}

    def _build_detail_row(self, meidai_meta: Dict[str, str], utterance_meta: Dict[str, str], tokens: Sequence[Token], idx: int, particle: str, pos_tag: str) -> Dict[str, Any]:
        speaker_id = utterance_meta.get("s", "")
        speaker_info = self.speaker_info.get(speaker_id, {})
        id_gender = self._gender_from_speaker_id(speaker_id)
        raw_gender = utterance_meta.get("sex", speaker_info.get("gender", "")) or ""
        return {
            "particle": particle,
            "pos_tag": pos_tag,
            "speaker": speaker_id,
            "gender": id_gender if id_gender != 'Unknown' else (raw_gender or "Unknown"),
            "age": utterance_meta.get("age", speaker_info.get("age", "Unknown")) or "Unknown",
            "conversation_id": meidai_meta.get("name", ""),
            "conversation_speakers": meidai_meta.get("speakers", ""),
            "context": self._preceding_context(tokens, idx),
            "example_sentence": self._example_sentence(tokens, idx),
        }

    # ==============================================================
    # 1. METADATA BRIDGE: Translating NUCC format into CEJC format
    # ==============================================================
    def analyze_sfp_by_partner(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Enrich main-speaker SFP rows with CEJC-formatted conversation partner characteristics."""
        if df_results.empty: return pd.DataFrame()

        def _format_age(age_str):
            if pd.isna(age_str) or str(age_str).strip() == "" or str(age_str).lower() == "unknown":
                return "Unknown"
                
            # Extract the first number found in the string (handles "22", "20代", etc.)
            match = re.search(r'\d+', str(age_str))
            if not match:
                return "Unknown"
                
            age_val = int(match.group())
            
            # Calculate the CEJC 5-year bucket (e.g., 22 -> 20-24)
            lower_bound = (age_val // 5) * 5
            upper_bound = lower_bound + 4
            
            return f"{lower_bound}-{upper_bound}歳"

        def _format_gender(g_str):
            # NUCC uses F/M, CEJC uses 女性/男性
            g = str(g_str).upper()
            if g in ['F', 'FEMALE', '女性']: return '女性'
            if g in ['M', 'MALE', '男性']: return '男性'
            return 'Unknown'

        conv_partners = {}
        for conv_id in df_results['conversation_id'].unique():
            speakers_str = df_results[df_results['conversation_id'] == conv_id]['conversation_speakers'].iloc[0]
            speakers = [s.strip() for s in str(speakers_str).split(',')]
            if len(speakers) == 2:
                conv_partners[conv_id] = speakers

        results = []
        for _, row in df_results.iterrows():
            conv_id = row['conversation_id']
            if conv_id not in conv_partners: continue
            
            main_spk = row['speaker']
            partner = next((s for s in conv_partners[conv_id] if s != main_spk), None)
            if not partner: continue

            p_info = self.speaker_info.get(partner, {})
            
            info = {
                'main_speaker_id': main_spk,
                'main_speaker_name': main_spk, # NUCC doesn't separate name/ID
                'main_gender': _format_gender(row['gender']),
                'main_age': _format_age(row['age']),
                'partner_speaker_id': partner,
                'partner_gender': _format_gender(p_info.get('gender')),
                'partner_age': _format_age(p_info.get('age')),
            }
            results.append({**row.to_dict(), **info})

        return pd.DataFrame(results)

    def count_all(
        self, target_sfps: Optional[Sequence[str]] = None, gender_filter: Optional[str] = None,
        age_filter: Optional[str] = None, conversation_filter: Optional[Sequence[str]] = None,
        include_compounds: bool = True, utterance_final_only: bool = True, verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], int]]:
        """Count particles and track NUCC word counts simultaneously."""

        all_results = []
        word_counts = defaultdict(int)
        conversation_filter_set = set(conversation_filter or [])

        if verbose: print("Streaming utterances from Meidai XML...")

        for index, (meidai_meta, utterance_meta, tokens) in enumerate(self._iter_utterances(), start=1):
            conversation_id = meidai_meta.get("name", "")
            speaker_id = utterance_meta.get("s", "")
            
            if not self._is_two_speaker_conversation(meidai_meta): continue
            if conversation_filter_set and conversation_id not in conversation_filter_set: continue

            # Accumulate word count for this speaker in this conversation
            content_token_count = sum(1 for t in tokens if self._is_content_token(t))
            word_counts[(speaker_id, conversation_id)] += content_token_count

            sfp_result = self.count_sfp_in_utterance(
                meidai_meta, utterance_meta, tokens, target_sfps=target_sfps,
                include_compounds=include_compounds, utterance_final_only=utterance_final_only,
            )
            all_results.extend(sfp_result["details"])

            if verbose and index % 5000 == 0:
                print(f"Processed {index:,} utterances...")

        return pd.DataFrame(all_results), dict(word_counts)

    @staticmethod
    def get_summary(df_results: pd.DataFrame) -> Dict[str, Any]:
        if df_results.empty: return {"total_count": 0, "by_particle": {}}
        return {
            "total_count": len(df_results),
            "by_particle": df_results["particle"].value_counts().to_dict(),
        }

    # ==============================================================
    # 2. CEJC HTML/EXCEL EXPORT METHODS (Unmodified)
    # ==============================================================

    @staticmethod
    def _normalize_by_word_count(
        df: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], row_col: str, col_col: str,
        speaker_col: str = 'speaker', conv_col: str = 'conversation_id', per_n: int = 1000,
    ) -> pd.DataFrame:
        counts = pd.crosstab(df[row_col], df[col_col])
        wc_by_group: Dict[Any, int] = {}
        for group_val, group_df in df.groupby(row_col):
            pairs = group_df[[speaker_col, conv_col]].drop_duplicates()
            total_wc = sum(word_counts.get((spk, conv), 0) for spk, conv in zip(pairs[speaker_col], pairs[conv_col]))
            wc_by_group[group_val] = max(total_wc, 1)
        wc_series = pd.Series(wc_by_group)
        return (counts.div(wc_series, axis=0) * per_n).round(2)

    @staticmethod
    def _html_page(title: str, accent: str, body: str) -> str:
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f0f2f5; }}
        h1 {{ color: #333; border-bottom: 3px solid {accent}; padding: 18px 24px 12px; margin: 0; background: white; font-size: 20px; }}
        h2 {{ color: #444; margin: 24px 0 8px; font-size: 16px; }}
        h3 {{ color: #666; margin: 20px 0 4px; font-size: 14px; }}
        table {{ border-collapse: collapse; margin: 8px 0 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.15); font-size: 13px; max-width: 100%; overflow-x: auto; display: block; }}
        th, td {{ border: 1px solid #e0e0e0; padding: 7px 12px; text-align: right; white-space: nowrap; color: #000; }}
        th:first-child, td:first-child {{ text-align: left; font-weight: 600; color: #000; background: #fafafa; position: sticky; left: 0; }}
        th {{ background: {accent}; color: white; font-weight: 600; }}
        tr:nth-child(even) td {{ background: #f9f9f9; }}
        tr:nth-child(even) td:first-child {{ background: #f3f3f3; }}
        tr:hover td {{ background: #eef4ff !important; }}
        .summary-box {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 16px 24px; background: white; margin-bottom: 4px; border-bottom: 1px solid #e0e0e0; }}
        .stat-item {{ text-align: center; min-width: 90px; }}
        .stat {{ font-size: 26px; color: {accent}; font-weight: 700; display: block; line-height: 1.1; }}
        .stat-label {{ font-size: 11px; color: #999; margin-top: 2px; }}
        .speaker-panel {{ padding: 20px 24px; display: block; }}
        .norm-note {{ font-size: 11px; color: #999; font-style: italic; margin: 2px 0 14px; }}
        .page-note {{ padding: 4px 24px 10px; color: #888; font-size: 12px; background: white; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
{body}
</body>
</html>'''

    def _render_norm_table(self, df: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], row_col: str, particles: List[str], row_order: Optional[List[str]] = None) -> str:
        if df.empty: return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
        try:
            norm = self._normalize_by_word_count(df, word_counts, row_col, 'particle')
            cols = [p for p in particles if p in norm.columns]
            if not cols: return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
            if row_order: norm = norm.reindex([r for r in row_order if r in norm.index])
            return norm[cols].to_html(float_format='%.2f', na_rep='-')
        except Exception as e: return f"<p style='color:red;font-size:13px;'>Error: {e}</p>\n"

    @staticmethod
    def _add_relative_age_col(df: pd.DataFrame) -> pd.DataFrame:
        def _rel(row):
            try:
                diff = (int(row['partner_age'].split('-')[0]) - int(row['main_age'].split('-')[0]))
                if diff < -RELATIVE_AGE_THRESHOLD: return '年下 (younger)'
                if diff > RELATIVE_AGE_THRESHOLD: return '年上 (older)'
                return '同年代 (same age)'
            except Exception: return 'Unknown'
        df = df.copy()
        df['relative_age'] = df.apply(_rel, axis=1)
        return df

    def _build_speaker_tab_html(self, spk_df: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], particles: List[str]) -> str:
        first = spk_df.iloc[0]
        name = first.get('main_speaker_name', '?')
        main_gender = first.get('main_gender', '?')
        main_age = first.get('main_age', '?')
        n_conv = spk_df['conversation_id'].nunique()
        n_sfp = len(spk_df)

        pairs = spk_df[['speaker', 'conversation_id']].drop_duplicates()
        n_words = sum(word_counts.get((spk, conv), 0) for spk, conv in zip(pairs['speaker'], pairs['conversation_id']))

        html_str = f'''<div class="summary-box">
    <div class="stat-item"><span class="stat">{n_conv}</span><span class="stat-label">会話数<br>(conversations)</span></div>
    <div class="stat-item"><span class="stat">{n_words:,}</span><span class="stat-label">総語数<br>(words spoken)</span></div>
    <div class="stat-item"><span class="stat">{n_sfp:,}</span><span class="stat-label">SFP総数<br>(SFPs used)</span></div>
    <div class="stat-item"><span class="stat">{main_gender}</span><span class="stat-label">性別<br>(gender)</span></div>
    <div class="stat-item"><span class="stat" style="font-size:18px;">{main_age}</span><span class="stat-label">年齢<br>(age)</span></div>
</div>
<div style="padding:8px 0 0;">
<p class="norm-note">すべての値 = {name} が 1,000 語あたりに使用する SFP 回数</p>
'''
        html_str += "<h3>相手の性別別 SFP 使用率 (SFP rate by partner gender)</h3>\n" + self._render_norm_table(spk_df, word_counts, 'partner_gender', particles)

        valid_age_df = spk_df[spk_df['partner_age'] != 'Unknown'].copy()
        html_str += "<h3>相手の年齢別 SFP 使用率 (SFP rate by partner age group)</h3>\n" + self._render_norm_table(valid_age_df, word_counts, 'partner_age', particles, row_order=AGE_ORDER)

        rel_df = self._add_relative_age_col(valid_age_df)
        rel_df = rel_df[rel_df['relative_age'] != 'Unknown']
        html_str += f"<h3>相手の相対年齢別 SFP 使用率 (SFP rate by partner relative age, ±{RELATIVE_AGE_THRESHOLD}歳基準)</h3>\n" + self._render_norm_table(rel_df, word_counts, 'relative_age', particles, row_order=['年下 (younger)', '同年代 (same age)', '年上 (older)'])

        if not rel_df.empty:
            rel_df = rel_df.copy()
            rel_df['gender_rel'] = rel_df['partner_gender'] + ' / ' + rel_df['relative_age']
            combo_order = [f'{g} / {r}' for g in ['女性', '男性'] for r in ['年下 (younger)', '同年代 (same age)', '年上 (older)']]
            html_str += "<h3>相手の性別 × 相対年齢別 SFP 使用率 (SFP rate by partner gender × relative age)</h3>\n" + self._render_norm_table(rel_df, word_counts, 'gender_rel', particles, row_order=combo_order)

        html_str += "</div>\n"
        return html_str

    def _export_html_tabbed(self, df_results: pd.DataFrame, word_counts: Dict[Tuple[str, str], int], output_path: Path, prefix: str):
        try:
            partner_data = self.analyze_sfp_by_partner(df_results)
            if partner_data.empty: return

            grouped = {sid: grp for sid, grp in partner_data.groupby('main_speaker_id')}
            if not grouped: return

            sorted_speakers = sorted(grouped.items(), key=lambda kv: kv[1].iloc[0].get('main_speaker_name', kv[0]))
            speaker_content_html = ''
            for sid, spk_df in sorted_speakers:
                name = spk_df.iloc[0].get('main_speaker_name', sid)
                content = self._build_speaker_tab_html(spk_df, word_counts, REPORT_PARTICLES)
                speaker_content_html += f'<div id="spk_{sid}" class="speaker-panel">\n<h2>{name} — SFP使用パターン（相手属性別）</h2>\n{content}\n</div>\n'

            n_speakers = len(sorted_speakers)
            body = f'<h1>SFP Analysis by Main Speaker — 話者別終助詞分析</h1>\n<p class="page-note">{n_speakers} 名の主話者 · 値 = 1,000 語あたり SFP 使用数</p>\n{speaker_content_html}'

            out_file = output_path / f'{prefix}_main_speaker_analysis.html'
            out_file.write_text(self._html_page('SFP Analysis by Main Speaker', '#4CAF50', body), encoding='utf-8')
            print(f"HTML file saved to: {out_file}")
        except Exception as e:
            print(f"Warning: Could not create HTML file: {e}")

    def _export_csv(self, df_results, output_path, prefix):
        df_results.to_csv(output_path / f'{prefix}_detailed.csv', index=False, encoding='utf-8-sig')
        particle_summary = df_results['particle'].value_counts().reset_index()
        particle_summary.columns = ['particle', 'count']
        particle_summary.to_csv(output_path / f'{prefix}_by_particle.csv', index=False, encoding='utf-8-sig')
        speaker_summary = df_results['speaker'].value_counts().reset_index()
        speaker_summary.columns = ['speaker', 'count']
        speaker_summary.to_csv(output_path / f'{prefix}_by_speaker.csv', index=False, encoding='utf-8-sig')
        print(f"CSV files saved to: {output_path}")

    def export_results(self, df_results: pd.DataFrame, output_dir: str, prefix: str = 'sfp_count', word_counts: Optional[Dict[Tuple[str, str], int]] = None, csv_output: bool = False):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        wc = word_counts or {}

        if csv_output: self._export_csv(df_results, output_path, prefix)
        self._export_html_tabbed(df_results, wc, output_path, prefix)


# ==============================================================
# 3. CEJC PLOTTING FUNCTIONS (Aligned with CEJC Updates)
# ==============================================================
def plot_adaptation_trends(counter_instance: SFPCounter, df_results: pd.DataFrame, word_counts: dict, output_dir: str):
    """Calculates SFP variance and plots the adaptation distribution across different opponent demographic groups. Includes Overall Male/Female aggregates."""
    if df_results.empty: return

    partner_df = counter_instance.analyze_sfp_by_partner(df_results)
    if partner_df.empty: return

    partner_df = counter_instance._add_relative_age_col(partner_df)
    partner_df = partner_df[partner_df['relative_age'] != 'Unknown']

    conv_stats = partner_df.groupby(
        ['main_speaker_name', 'main_gender', 'main_age', 'speaker', 'conversation_id', 'partner_gender', 'relative_age']
    ).size().reset_index(name='sfp_count')

    conv_stats['word_count'] = conv_stats.apply(lambda row: word_counts.get((row['speaker'], row['conversation_id']), 1), axis=1)
    conv_stats['word_count'] = conv_stats['word_count'].replace(0, 1)
    conv_stats['sfp_rate'] = (conv_stats['sfp_count'] / conv_stats['word_count']) * 1000

    # 3. Create Custom Display Names (Name + Age + Gender)
    conv_stats['display_name'] = conv_stats.apply(lambda row: f"{row['main_speaker_name']} ({row['main_age']}, {row['main_gender']})", axis=1)

    # --- Create Overall Aggregate Data ---
    male_data = conv_stats[conv_stats['main_gender'].isin(['男性', 'M', 'Male'])].copy()
    male_data['display_name'] = 'Overall Male (男性 全体)'
    male_data['main_speaker_name'] = '0_Overall_Male'
    
    female_data = conv_stats[conv_stats['main_gender'].isin(['女性', 'F', 'Female'])].copy()
    female_data['display_name'] = 'Overall Female (女性 全体)'
    female_data['main_speaker_name'] = '1_Overall_Female'

    conv_stats = pd.concat([male_data, female_data, conv_stats], ignore_index=True)
    # ----------------------------------------

    def gender_sort_key(gender_str, name_str):
        if '0_Overall_Male' in name_str: return -2
        if '1_Overall_Female' in name_str: return -1
        
        g = str(gender_str).lower()
        if g in ['男性', 'm', 'male']: return 0
        if g in ['女性', 'f', 'female']: return 1
        return 2

    unique_speakers = conv_stats[['display_name', 'main_gender', 'main_speaker_name']].drop_duplicates()
    unique_speakers['sort_rank'] = unique_speakers.apply(
        lambda row: gender_sort_key(row['main_gender'], row['main_speaker_name']), axis=1
    )
    unique_speakers = unique_speakers.sort_values(by=['sort_rank', 'main_speaker_name'])
    speaker_order = unique_speakers['display_name'].tolist()

    variance_stats = conv_stats.groupby('display_name')['sfp_rate'].var().fillna(0).reset_index()
    variance_stats.rename(columns={'sfp_rate': 'sfp_variance'}, inplace=True)
    variance_stats.sort_values(by='sfp_variance', ascending=False, inplace=True)
    variance_stats.to_csv(Path(output_dir) / 'sfp_adaptation_variance.csv', index=False, encoding='utf-8-sig')

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    age_order = ['年下 (younger)', '同年代 (same age)', '年上 (older)']
    hue_order = ['男性', '女性']  # <--- 1. ADD THIS VARIABLE
    
    box_palette = {'女性': '#e74c3c', 'F': '#e74c3c', 'Female': '#e74c3c', '男性': '#3498db', 'M': '#3498db', 'Male': '#3498db'}
    dot_palette = {'女性': '#2ecc71', 'F': '#2ecc71', 'Female': '#2ecc71', '男性': '#f39c12', 'M': '#f39c12', 'Male': '#f39c12'}

    g = sns.catplot(
        data=conv_stats, x='relative_age', y='sfp_rate', hue='partner_gender', 
        hue_order=hue_order,  # <--- 2. ADD THIS PARAMETER
        col='display_name',
        col_wrap=3, col_order=speaker_order, kind='box', order=age_order, height=4, aspect=1.2,
        sharey=True, palette=box_palette
    )

    g.map_dataframe(
        sns.stripplot, x='relative_age', y='sfp_rate', hue='partner_gender',
        hue_order=hue_order,  # <--- 3. ADD THIS PARAMETER
        order=age_order, dodge=True, alpha=0.8, palette=dot_palette
    )

    g.set_axis_labels("相手の相対年齢 (Opponent Relative Age)", "SFP Rate (per 1,000 words)")
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle('SFP Usage Adaptation by Main Speaker (Opponent Age & Gender)', fontsize=16)

    plt.savefig(Path(output_dir) / 'sfp_adaptation_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_adaptation_trends_by_particle(counter_instance, df_results: pd.DataFrame, word_counts: dict, output_dir: str, target_particles=['ね', 'よ', 'な', 'わ', 'よね']):
    """Calculates SFP variance by specific particle type and plots the adaptation distribution."""
    if df_results.empty: return

    partner_df = counter_instance.analyze_sfp_by_partner(df_results)
    if partner_df.empty: return

    partner_df = counter_instance._add_relative_age_col(partner_df)
    partner_df = partner_df[partner_df['relative_age'] != 'Unknown']

    conv_stats = partner_df.groupby(
        ['main_speaker_name', 'main_gender', 'main_age', 'speaker', 'conversation_id', 'partner_gender', 'relative_age', 'particle']
    ).size().reset_index(name='sfp_count')

    conv_stats['word_count'] = conv_stats.apply(lambda row: word_counts.get((row['speaker'], row['conversation_id']), 1), axis=1)
    conv_stats['word_count'] = conv_stats['word_count'].replace(0, 1)
    conv_stats['sfp_rate'] = (conv_stats['sfp_count'] / conv_stats['word_count']) * 1000

    conv_stats['display_name'] = conv_stats.apply(lambda row: f"{row['main_speaker_name']} ({row['main_age']}, {row['main_gender']})", axis=1)

    # --- Create Overall Aggregate Data ---
    male_data = conv_stats[conv_stats['main_gender'].isin(['男性', 'M', 'Male'])].copy()
    male_data['display_name'] = 'Overall Male (男性 全体)'
    male_data['main_speaker_name'] = '0_Overall_Male'
    
    female_data = conv_stats[conv_stats['main_gender'].isin(['女性', 'F', 'Female'])].copy()
    female_data['display_name'] = 'Overall Female (女性 全体)'
    female_data['main_speaker_name'] = '1_Overall_Female'

    conv_stats = pd.concat([male_data, female_data, conv_stats], ignore_index=True)
    # ----------------------------------------

    def gender_sort_key(gender_str, name_str):
        if '0_Overall_Male' in name_str: return -2
        if '1_Overall_Female' in name_str: return -1
        g = str(gender_str).lower()
        if g in ['男性', 'm', 'male']: return 0
        if g in ['女性', 'f', 'female']: return 1
        return 2

    unique_speakers = conv_stats[['display_name', 'main_gender', 'main_speaker_name']].drop_duplicates()
    unique_speakers['sort_rank'] = unique_speakers.apply(lambda row: gender_sort_key(row['main_gender'], row['main_speaker_name']), axis=1)
    unique_speakers = unique_speakers.sort_values(by=['sort_rank', 'main_speaker_name'])
    speaker_order = unique_speakers['display_name'].tolist()

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    age_order = ['年下 (younger)', '同年代 (same age)', '年上 (older)']
    box_palette = {'女性': '#e74c3c', 'F': '#e74c3c', 'Female': '#e74c3c', '男性': '#3498db', 'M': '#3498db', 'Male': '#3498db'}
    dot_palette = {'女性': '#2ecc71', 'F': '#2ecc71', 'Female': '#2ecc71', '男性': '#f39c12', 'M': '#f39c12', 'Male': '#f39c12'}

    for particle in target_particles:
        part_stats = conv_stats[conv_stats['particle'] == particle]
        if part_stats.empty: continue

        g = sns.catplot(
            data=part_stats, x='relative_age', y='sfp_rate', hue='partner_gender', col='display_name',
            col_wrap=3, col_order=speaker_order, kind='box', order=age_order, height=4, aspect=1.2,
            sharey=False, palette=box_palette
        )

        g.map_dataframe(
            sns.stripplot, x='relative_age', y='sfp_rate', hue='partner_gender',
            order=age_order, dodge=True, alpha=0.8, palette=dot_palette
        )

        g.set_axis_labels("相手の相対年齢 (Opponent Relative Age)", f"'{particle}' Rate (per 1,000 words)")
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.92)
        g.fig.suptitle(f"'{particle}' Usage Adaptation by Main Speaker", fontsize=16)

        plot_path = Path(output_dir) / f'sfp_adaptation_distribution_{particle}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved particle graph: {plot_path.name}")


def plot_sfp_composition_by_opponent(counter_instance, df_results: pd.DataFrame, output_dir: str):
    """Generates a small-multiples line chart showing SFP composition shifts by opponent type."""
    if df_results.empty: return

    import numpy as np

    partner_df = counter_instance.analyze_sfp_by_partner(df_results)
    if partner_df.empty: return

    partner_df = counter_instance._add_relative_age_col(partner_df)
    partner_df = partner_df[partner_df['relative_age'] != 'Unknown']

    top_5_sfps = partner_df['particle'].value_counts().head(5).index.tolist()
    print(f"Top 5 SFPs selected for small-multiples composition analysis: {top_5_sfps}")

    filtered_df = partner_df[partner_df['particle'].isin(top_5_sfps)].copy()

    rel_age_map = {
        '年下 (younger)': 'Younger',
        '同年代 (same age)': 'Same Age',
        '年上 (older)': 'Older',
    }
    filtered_df['rel_age_short'] = filtered_df['relative_age'].map(rel_age_map)

    male_gender_vals   = ['男性', 'M', 'Male']
    female_gender_vals = ['女性', 'F', 'Female']

    opponent_labels = ['M/Younger', 'M/Same Age', 'M/Older', 'F/Younger', 'F/Same Age', 'F/Older']

    def compute_pct(df, main_gender_vals, partner_gender_vals, rel_ages):
        sub = df[
            df['main_gender'].isin(main_gender_vals) &
            df['partner_gender'].isin(partner_gender_vals) &
            df['rel_age_short'].isin(rel_ages)
        ]
        counts = sub['particle'].value_counts()
        total = counts.sum()
        if total == 0:
            return {p: 0.0 for p in top_5_sfps}
        return {p: (counts.get(p, 0) / total * 100) for p in top_5_sfps}

    speaker_groups = [
        ('Male Speaker', male_gender_vals),
        ('Female Speaker', female_gender_vals),
    ]

    opponent_configs = [
        (male_gender_vals,   ['Younger']),
        (male_gender_vals,   ['Same Age']),
        (male_gender_vals,   ['Older']),
        (female_gender_vals, ['Younger']),
        (female_gender_vals, ['Same Age']),
        (female_gender_vals, ['Older']),
    ]

    data = {}
    for spk_label, spk_genders in speaker_groups:
        data[spk_label] = {}
        for sfp in top_5_sfps:
            data[spk_label][sfp] = []
            for partner_genders, rel_ages in opponent_configs:
                pct = compute_pct(filtered_df, spk_genders, partner_genders, rel_ages)
                data[spk_label][sfp].append(pct[sfp])

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    fig.suptitle('SFP Usage Shifts by Opponent Type', fontsize=18, y=1.05)

    male_color   = '#1f77b4'
    female_color = '#ff7f0e'

    for i, sfp in enumerate(top_5_sfps):
        ax = axes[i]

        ax.plot(opponent_labels, [np.nan] * len(opponent_labels))

        male_vals   = data['Male Speaker'][sfp]
        female_vals = data['Female Speaker'][sfp]

        ax.plot(opponent_labels[:3], male_vals[:3],   marker='o', color=male_color,   linewidth=2.5,
                label='Male Speaker'   if i == 0 else '')
        ax.plot(opponent_labels[3:],  male_vals[3:],   marker='o', color=male_color,   linewidth=2.5)

        ax.plot(opponent_labels[:3], female_vals[:3], marker='s', color=female_color, linewidth=2.5,
                linestyle='--', label='Female Speaker' if i == 0 else '')
        ax.plot(opponent_labels[3:],  female_vals[3:],  marker='s', color=female_color, linewidth=2.5,
                linestyle='--')

        ax.set_title(f'SFP: {sfp}', fontsize=14)
        ax.set_xticks(range(len(opponent_labels)))
        ax.set_xticklabels(opponent_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_ylabel('Percentage Share (%)', fontsize=12)

    fig.legend(loc='upper right', bbox_to_anchor=(0.98, 1.05), fontsize=12)

    plt.tight_layout()
    plot_path = Path(output_dir) / 'sfp_top5_composition_grouped.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved small-multiples composition chart: {plot_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count sentence-final particles in the Meidai corpus (NUCC/Himawari).")
    parser.add_argument("--meidai-root", type=str, default="/home/ryuu/Ryu/Corpus/nucc/Himawari_nucc/Himawari_meidai")
    parser.add_argument("--output-dir", type=str, default="/home/ryuu/Ryu/Dataset_Analysis/nucc_analysis/output")
    parser.add_argument("--csv-output", action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("Meidai Sentence Final Particle Counter (Aligned with CEJC)")
    print("=" * 60)

    counter = SFPCounter(args.meidai_root)

    print("\nCounting SFPs...")
    df_results, word_counts = counter.count_all(verbose=True)

    summary = counter.get_summary(df_results)
    print(f"\nTotal SFP occurrences: {summary['total_count']}")
    
    counter.export_results(df_results, args.output_dir, prefix="meidai_sfp", word_counts=word_counts, csv_output=args.csv_output)
    
    print("\nGenerating adaptation distribution graphs...")
    # 1. Original graph (all combined)
    plot_adaptation_trends(counter, df_results, word_counts, args.output_dir)
    
    # 2. Graph per specific particle
    target_particles_to_plot = ['ね', 'よ', 'な', 'わ', 'よね', 'さ', 'の']
    plot_adaptation_trends_by_particle(counter, df_results, word_counts, args.output_dir, target_particles_to_plot)
    
    # 3. Proportional Composition Chart
    plot_sfp_composition_by_opponent(counter, df_results, args.output_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()