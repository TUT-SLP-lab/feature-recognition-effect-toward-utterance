#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CEJC Sentence Final Particle (SFP) Counter
終助詞カウンター

Counts sentence-final particles at utterance ends in the CEJC corpus.
"""

import ast
import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

#import for distribution calculation
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# ==============================================================
# PATHS & ENCODING
# ==============================================================
CEJC_ROOT     = '/home/ryuu/Ryu/Corpus/CEJC'
OUTPUT_DIR    = '/home/ryuu/Ryu/Dataset_Analysis/CEJC_SFP_Analyze/output'
FILE_ENCODING = 'cp932'   # Shift-JIS; change to 'utf-8' if corpus is re-encoded

# ==============================================================
# ANALYSIS FLAGS
# ==============================================================
UTTERANCE_FINAL_ONLY = True   # Only count SFPs at utterance end
INCLUDE_COMPOUNDS    = True   # Count compound particles like よね
TWO_SPEAKER_ONLY     = True   # Restrict to 2-person conversations
INCLUDE_TOKA         = False  # Include とか (via PRECEDING_COMPOUND_RULES)

# ==============================================================
# CORPUS COLUMN NAMES  ← update here if CSV format changes
# ==============================================================
COL_POS           = '品詞'
COL_LEMMA         = '語彙素'
COL_SURFACE       = '書字形'
COL_SPEAKER       = '話者ラベル'
COL_UTT_START     = '発話単位の開始時刻'
COL_UTT_END       = '発話単位の終了時刻'
COL_SENTENCE_FLAG = '文頭フラグ'
COL_CONV_ID       = '会話ID'
COL_SPEAKER_COUNT = '話者数'
COL_SPEAKER_ID    = '話者ID'
COL_SPEAKER_NAME  = '話者名'
COL_GENDER        = '性別'
COL_AGE           = '年齢'

# ==============================================================
# POS TAGS
# ==============================================================
POS_SENTENCE_FINAL = '助詞-終助詞'
POS_ADVERBIAL      = '助詞-副助詞'
POS_CASE           = '助詞-格助詞'

# ==============================================================
# SFP MODE — select which SFP taxonomy to use at runtime
# "old" = original set (default); "new" = gendered taxonomy from the 3-column table
# ==============================================================
SFP_MODE = 'old'   # overridden by --sfp-mode CLI flag

# ==============================================================
# OLD SFP TARGETS  ← original set
# ==============================================================
TARGET_SFPS = [
    'よ', 'ね', 'か', 'かい', 'さ', 'じゃん', 'な',
    'の', 'わ', 'もの', 'もん', 'っけ', 'かしら', 'や',
]

# Compound SFPs: two consecutive SFP tokens treated as one particle.
# Format: (first_surface, second_surface, result_label)
# Condition: both tokens must be POS_SENTENCE_FINAL; second must be utterance-final.
COMPOUND_RULES = [
    ('よ', 'ね', 'よね'),
]

# Preceding-token compounds: particle formed together with the token before it.
# Format: (prev_surface, prev_pos, curr_surface, curr_pos, result_label)
# Used when INCLUDE_TOKA is True (or --include-toka CLI flag).
PRECEDING_COMPOUND_RULES = [
    ('と', POS_CASE, 'か', POS_ADVERBIAL, 'とか'),
]

# ==============================================================
# REPORT DISPLAY  ← particles shown in HTML/Excel tables, in order
# ==============================================================
REPORT_PARTICLES = [
    'ね', 'よ', 'か', 'よね', 'の', 'な', 'さ', 'じゃん', 'っけ', 'もん', 'わ', 'や', 'やん',
]

# ==============================================================
# NEW SFP TARGETS  ← gendered taxonomy (algorithm-detectable subset)
#
# NOTE — items that REQUIRE LLM / rich context and are NOT covered here:
#   Feminine:  だわ/だったわ, だ/だった+わね/わよ/わよね (preceding だ/だった needed),
#              名詞+よ / 形容動詞+よ (preceding POS), 形容動詞+の / 名詞+の (preceding POS),
#              名詞+ね / 形容詞+ね (preceding POS), 動詞終止形+の / イ形容詞+の (preceding POS),
#              て形+ね (preceding verb form)
#   Neutral:   動詞・イ形容詞終止形 / 名詞・形容動詞語幹 (context constraints, not particles),
#              わ (イントネーション上がらない) (prosody), て形(依頼) (semantic intent),
#              感嘆詞・定型句 (semantic category)
#   Masculine: 動詞命令形+よ (preceding verb form), 否定命令のな/なよ (neg-command context),
#              同意を求めるな / na-agreement (pragmatic), 動詞+よ / イ形容詞+よ (preceding POS),
#              名詞+だ / 形容動詞+だ (preceding POS)
# ==============================================================
NEW_TARGET_SFPS = [
    # Feminine
    'わ', 'かしら', 'や',
    # Neutral
    'よね', 'もん', 'じゃない', 'じゃん', 'かな', 'だって', 'って', 'ね',
    # Masculine
    'ぞ', 'ぜ', 'ええ', 'か', 'さ', 'け', 'わな',
    # Shared base particles (compound targets below)
    'よ', 'な',
]

NEW_COMPOUND_RULES = [
    # Feminine compounds
    ('わ', 'ね',  'わね'),
    ('わ', 'よ',  'わよ'),
    ('よ', 'ね',  'よね'),   # also neutral
    ('の', 'ね',  'のね'),
    ('の', 'よ',  'のよ'),
    ('の', 'よね','のよね'),
    ('も', 'ん',  'もんね'),  # もん+ね handled via surface
    # Masculine compounds
    ('か', 'よ',  'かよ'),
    ('だ', 'よ',  'だよ'),
    ('だ', 'ね',  'だね'),
    ('ん', 'だ',  'んだ'),
    ('わ', 'な',  'わな'),
    ('っ', 'す',  'っすよ'),  # partial; full form matched via surface below
]

NEW_REPORT_PARTICLES = [
    # Feminine
    'わ', 'わね', 'わよ', 'かしら', 'や',
    # Neutral
    'よね', 'もん', 'じゃない', 'じゃん', 'かな', 'だって', 'って', 'ね',
    # Masculine
    'ぞ', 'ぜ', 'かよ', 'だよ', 'だね', 'だよね', 'んだ', 'んだね', 'んだよ', 'んだよね',
    'だろ', 'だろう', 'おうか', 'か', 'け', 'わな', 'だろうな', 'だな', 'っすよ', 'っすよね', 'さ',
    # Shared
    'よ', 'な',
]

AGE_ORDER = [
    '0-4歳',  '5-9歳',  '10-14歳', '15-19歳', '20-24歳', '25-29歳',
    '30-34歳', '35-39歳', '40-44歳', '45-49歳', '50-54歳', '55-59歳',
    '60-64歳', '65-69歳', '70-74歳', '75-79歳', '80-84歳', '85-89歳',
    '90-94歳', '95-99歳',
]

RELATIVE_AGE_THRESHOLD   = 5   # years; partner is "younger/older" if diff >= this
CONTEXT_BEFORE           = 8   # tokens shown before the SFP in example sentences
CONTEXT_AFTER            = 2   # tokens shown after the SFP in example sentences
MIN_SPEAKER_CONVERSATIONS = 3  # min conversations for a main speaker to appear as an HTML tab


class SFPCounter:
    """Sentence Final Particle Counter for CEJC corpus."""

    def __init__(self, cejc_root: str = CEJC_ROOT, encoding: str = FILE_ENCODING):
        self.cejc_root = Path(cejc_root)
        self.encoding  = encoding
        self.speaker_data      = self._load_csv('Speaker_data.csv')
        self.conversation_data = self._load_csv('Conversation.csv')

    # ----------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------

    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = self.cejc_root / filename
        if not path.exists():
            return pd.DataFrame()
        for enc in ('utf-8', self.encoding):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.DataFrame()

    def _load_morph_file(self, filepath: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath, encoding=self.encoding)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # Speaker helpers
    # ----------------------------------------------------------

    @staticmethod
    def _extract_speaker_name(label: str) -> str:
        """'IC01_玲子' → '玲子'  (everything after the first underscore)."""
        parts = label.split('_')
        return '_'.join(parts[1:]) if len(parts) >= 2 else label

    def _build_speaker_lookup(self) -> tuple:
        """Return (name→gender, name→age) dicts built from speaker metadata."""
        name_to_gender, name_to_age = {}, {}
        for _, row in self.speaker_data.iterrows():
            name = row.get(COL_SPEAKER_NAME, '')
            if pd.notna(name):
                if pd.notna(row.get(COL_GENDER)):
                    name_to_gender[name] = row[COL_GENDER]
                if pd.notna(row.get(COL_AGE)):
                    name_to_age[name] = row[COL_AGE]
        return name_to_gender, name_to_age

    # ----------------------------------------------------------
    # Utterance boundary detection
    # ----------------------------------------------------------

    def _is_utterance_final(self, df: pd.DataFrame, idx: int) -> bool:
        if idx >= len(df) - 1:
            return True
        curr = df.iloc[idx]
        nxt  = df.iloc[idx + 1]
        return nxt[COL_SENTENCE_FLAG] == 'B' or curr[COL_UTT_END] != nxt[COL_UTT_END]

    # ----------------------------------------------------------
    # Compound particle detection (driven by the RULES constants)
    # ----------------------------------------------------------

    def _detect_compound_sfp(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """Return compound label if the token at idx starts a COMPOUND_RULES match."""
        if idx >= len(df) - 1:
            return None
        curr = df.iloc[idx]
        nxt  = df.iloc[idx + 1]
        for s1, s2, label in COMPOUND_RULES:
            if (curr[COL_SURFACE] == s1 and curr[COL_POS] == POS_SENTENCE_FINAL and
                    nxt[COL_SURFACE]  == s2 and nxt[COL_POS]  == POS_SENTENCE_FINAL and
                    self._is_utterance_final(df, idx + 1)):
                return label
        return None

    def _detect_preceding_compound(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """Return compound label if token at idx completes a PRECEDING_COMPOUND_RULES match."""
        if idx < 1:
            return None
        prev = df.iloc[idx - 1]
        curr = df.iloc[idx]
        for ps, pp, cs, cp, label in PRECEDING_COMPOUND_RULES:
            if (prev[COL_SURFACE] == ps and prev[COL_POS] == pp and
                    curr[COL_SURFACE] == cs and curr[COL_POS] == cp):
                return label
        return None

    # ----------------------------------------------------------
    # Context & example sentence helpers
    # ----------------------------------------------------------

    def _get_preceding_context(self, df: pd.DataFrame, idx: int, n: int = 3) -> List[str]:
        return df.iloc[max(0, idx - n):idx][COL_SURFACE].tolist()

    def _get_example_sentence(self, df: pd.DataFrame, idx: int) -> str:
        start  = max(0, idx - CONTEXT_BEFORE)
        end    = min(len(df), idx + CONTEXT_AFTER + 1)
        tokens = df.iloc[start:end][COL_SURFACE].tolist()
        sfp_pos = idx - start
        if 0 <= sfp_pos < len(tokens):
            tokens[sfp_pos] = f"【{tokens[sfp_pos]}】"
        return ''.join(tokens)

    def _build_detail_row(
        self, df: pd.DataFrame, idx: int, particle: str, filepath: Path,
        lemma: Optional[str] = None,
    ) -> dict:
        row = df.iloc[idx]
        return {
            'particle':         particle,
            'lemma':            lemma if lemma is not None else row[COL_LEMMA],
            'pos_tag':          row[COL_POS],
            'speaker':          row[COL_SPEAKER],
            'start_time':       row[COL_UTT_START],
            'end_time':         row[COL_UTT_END],
            'context':          self._get_preceding_context(df, idx),
            'example_sentence': self._get_example_sentence(df, idx),
            'file':             filepath.name,
            'conversation_id':  row[COL_CONV_ID],
        }

    @staticmethod
    def _get_example_text(row) -> str:
        """Return display text from a result row (prefers example_sentence column)."""
        text = row.get('example_sentence', '')
        if text and not pd.isna(text):
            return str(text)
        ctx = row.get('context', [])
        if isinstance(ctx, str):
            try:
                ctx = ast.literal_eval(ctx)
            except Exception:
                ctx = []
        return ''.join(ctx) + f"【{row.get('particle', '')}】"

    # ----------------------------------------------------------
    # Counting
    # ----------------------------------------------------------

    def _count_speakers(self, df: pd.DataFrame) -> int:
        if df.empty or COL_SPEAKER not in df.columns:
            return 0
        return df[COL_SPEAKER].nunique()

    def _get_two_speaker_conversations(self) -> List[str]:
        if self.conversation_data.empty or COL_SPEAKER_COUNT not in self.conversation_data.columns:
            return []
        return self.conversation_data[
            self.conversation_data[COL_SPEAKER_COUNT] == 2
        ][COL_CONV_ID].tolist()

    def count_sfp_in_file(
        self,
        filepath: Path,
        target_sfps: Optional[List[str]] = None,
        include_compounds: bool = INCLUDE_COMPOUNDS,
        utterance_final_only: bool = UTTERANCE_FINAL_ONLY,
        include_preceding_compounds: bool = INCLUDE_TOKA,
    ) -> Dict[str, Any]:
        """Count SFPs (and optionally preceding compounds like とか) in one morph file."""
        df = self._load_morph_file(filepath)
        if df.empty:
            return {
                'file': str(filepath), 'total': 0,
                'counts': {}, 'details': [], 'word_counts': {},
            }

        # Count total tokens per speaker for word-count normalization
        word_counts: Dict[str, int] = {}
        if COL_SPEAKER in df.columns:
            for spk_label, grp in df.groupby(COL_SPEAKER):
                word_counts[str(spk_label)] = len(grp)

        counts    = defaultdict(int)
        details   = []
        processed = set()

        # --- SFP tokens (終助詞) ---
        for idx in df.index[df[COL_POS] == POS_SENTENCE_FINAL].tolist():
            if idx in processed:
                continue

            if include_compounds:
                label = self._detect_compound_sfp(df, idx)
                if label:
                    if target_sfps is None or label in target_sfps:
                        counts[label] += 1
                        details.append(
                            self._build_detail_row(df, idx, label, filepath, lemma=label))
                    processed.update({idx, idx + 1})
                    continue

            if utterance_final_only and not self._is_utterance_final(df, idx):
                continue

            row      = df.iloc[idx]
            particle = row[COL_SURFACE]
            lemma    = row[COL_LEMMA]

            # Strict boundary check: the surface form must exactly match a target SFP,
            # OR the lemma matches and the surface is ≤3 chars (catches short colloquial
            # variants like もん whose lemma is もの, while blocking long mis-tagged forms).
            if target_sfps is not None:
                if particle in target_sfps:
                    pass
                elif lemma in target_sfps and len(particle) <= 3:
                    pass
                else:
                    continue

            counts[particle] += 1
            details.append(self._build_detail_row(df, idx, particle, filepath))
            processed.add(idx)

        # --- Preceding-compound tokens (e.g., とか via 副助詞) ---
        if include_preceding_compounds:
            for idx in df.index[df[COL_POS] == POS_ADVERBIAL].tolist():
                if utterance_final_only and not self._is_utterance_final(df, idx):
                    continue
                label = self._detect_preceding_compound(df, idx)
                if label is None:
                    continue
                counts[label] += 1
                details.append(self._build_detail_row(df, idx, label, filepath))

        return {
            'file':        str(filepath),
            'total':       sum(counts.values()),
            'counts':      dict(counts),
            'details':     details,
            'word_counts': word_counts,
        }

    def count_all(
        self,
        target_sfps: Optional[List[str]] = None,
        include_toka: bool = INCLUDE_TOKA,
        gender_filter: Optional[str] = None,
        age_filter: Optional[str] = None,
        conversation_filter: Optional[List[str]] = None,
        file_pattern: str = '**/*-morphSUW.csv',
        include_compounds: bool = INCLUDE_COMPOUNDS,
        utterance_final_only: bool = UTTERANCE_FINAL_ONLY,
        two_speaker_only: bool = TWO_SPEAKER_ONLY,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], int]]:
        """Count SFPs across all morph files, with optional filtering.

        Returns (df_results, word_counts) where word_counts maps
        (speaker_label, conv_id) → total token count for that speaker in that conversation.
        """
        morph_files = list(self.cejc_root.glob(file_pattern))

        if conversation_filter:
            morph_files = [f for f in morph_files
                           if any(c in f.name for c in conversation_filter)]

        if verbose:
            print(f"Found {len(morph_files)} morphological analysis files")

        if two_speaker_only:
            morph_files = self._filter_two_speaker_files(morph_files, verbose)

        all_details: List[dict] = []
        word_counts: Dict[Tuple[str, str], int] = {}

        for i, filepath in enumerate(morph_files):
            if verbose and (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(morph_files)}: {filepath.name}")
            result = self.count_sfp_in_file(
                filepath,
                target_sfps=target_sfps,
                include_compounds=include_compounds,
                utterance_final_only=utterance_final_only,
                include_preceding_compounds=include_toka,
            )
            all_details.extend(result['details'])

            # Accumulate per-(speaker, conversation) word counts
            conv_id = filepath.stem.replace('-morphSUW', '')
            for spk, wc in result.get('word_counts', {}).items():
                word_counts[(spk, conv_id)] = wc

        if not all_details:
            return pd.DataFrame(), word_counts

        df_results = pd.DataFrame(all_details)

        if not self.speaker_data.empty and (gender_filter or age_filter):
            df_results = self._apply_speaker_filters(df_results, gender_filter, age_filter)

        return df_results, word_counts

    def _filter_two_speaker_files(self, morph_files: List[Path], verbose: bool) -> List[Path]:
        if verbose:
            print("Filtering for conversations with exactly 2 speakers...")
        two_speaker_ids = self._get_two_speaker_conversations()
        if two_speaker_ids:
            filtered = [f for f in morph_files
                        if f.stem.replace('-morphSUW', '') in two_speaker_ids]
        else:
            if verbose:
                print("Warning: No conversation metadata found, counting speakers from files...")
            filtered = [f for f in morph_files
                        if self._count_speakers(self._load_morph_file(f)) == 2]
        if verbose:
            print(f"Found {len(filtered)} files with exactly 2 speakers")
        return filtered

    def _apply_speaker_filters(
        self,
        df: pd.DataFrame,
        gender_filter: Optional[str],
        age_filter: Optional[str],
    ) -> pd.DataFrame:
        if df.empty or COL_SPEAKER_NAME not in self.speaker_data.columns:
            return df
        name_to_gender, name_to_age = self._build_speaker_lookup()

        def should_include(label):
            name = self._extract_speaker_name(label)
            if gender_filter and name_to_gender.get(name) != gender_filter:
                return False
            if age_filter and name_to_age.get(name) != age_filter:
                return False
            return True

        return df[df['speaker'].apply(should_include)]

    # ----------------------------------------------------------
    # Summary & metadata enrichment
    # ----------------------------------------------------------

    def get_summary(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        if df_results.empty:
            return {'total_count': 0, 'by_particle': {}, 'by_speaker': {}, 'by_conversation': {}}
        return {
            'total_count':     len(df_results),
            'by_particle':     df_results['particle'].value_counts().to_dict(),
            'by_speaker':      df_results['speaker'].value_counts().to_dict(),
            'by_conversation': df_results['conversation_id'].value_counts().to_dict(),
        }

    def _add_speaker_metadata(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Add 'gender' and 'age' columns from speaker metadata."""
        if df_results.empty or self.speaker_data.empty:
            return df_results
        name_to_gender, name_to_age = self._build_speaker_lookup()
        df = df_results.copy()
        df['gender'] = df['speaker'].apply(
            lambda l: name_to_gender.get(self._extract_speaker_name(l), 'Unknown'))
        df['age'] = df['speaker'].apply(
            lambda l: name_to_age.get(self._extract_speaker_name(l), 'Unknown'))
        return df

    # ----------------------------------------------------------
    # Normalization
    # ----------------------------------------------------------

    @staticmethod
    def _normalize_by_word_count(
        df: pd.DataFrame,
        word_counts: Dict[Tuple[str, str], int],
        row_col: str,
        col_col: str,
        speaker_col: str = 'speaker',
        conv_col: str = 'conversation_id',
        per_n: int = 1000,
    ) -> pd.DataFrame:
        """Crosstab(row_col × col_col) normalized as SFP occurrences per per_n words spoken.

        For each row group, sums the word counts of the relevant speaker across
        the conversations that belong to that group, then divides accordingly.
        """
        counts = pd.crosstab(df[row_col], df[col_col])
        wc_by_group: Dict[Any, int] = {}
        for group_val, group_df in df.groupby(row_col):
            pairs = group_df[[speaker_col, conv_col]].drop_duplicates()
            total_wc = sum(
                word_counts.get((spk, conv), 0)
                for spk, conv in zip(pairs[speaker_col], pairs[conv_col])
            )
            wc_by_group[group_val] = max(total_wc, 1)  # avoid division by zero
        wc_series = pd.Series(wc_by_group)
        return (counts.div(wc_series, axis=0) * per_n).round(2)

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_results(
        self,
        df_results: pd.DataFrame,
        output_dir: str,
        prefix: str = 'sfp_count',
        word_counts: Optional[Dict[Tuple[str, str], int]] = None,
        csv_output: bool = False,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        enriched = self._add_speaker_metadata(df_results)
        wc = word_counts or {}

        if csv_output:
            self._export_csv(df_results, enriched, output_path, prefix)
        else:
            print("CSV output disabled. Use --csv-output to generate CSV files.")
        self._export_excel(df_results, enriched, output_path, prefix)
        self._export_html_tabbed(df_results, wc, output_path, prefix)

    def _export_csv(self, df_results, enriched, output_path, prefix):
        df_results.to_csv(
            output_path / f'{prefix}_detailed.csv', index=False, encoding='utf-8-sig')

        particle_summary = df_results['particle'].value_counts().reset_index()
        particle_summary.columns = ['particle', 'count']
        particle_summary.to_csv(
            output_path / f'{prefix}_by_particle.csv', index=False, encoding='utf-8-sig')

        speaker_summary = df_results['speaker'].value_counts().reset_index()
        speaker_summary.columns = ['speaker', 'count']
        speaker_summary.to_csv(
            output_path / f'{prefix}_by_speaker.csv', index=False, encoding='utf-8-sig')

        pd.crosstab(df_results['speaker'], df_results['particle']).to_csv(
            output_path / f'{prefix}_crosstab.csv', encoding='utf-8-sig')

        if 'gender' in enriched.columns:
            pd.crosstab(enriched['gender'], enriched['particle'], margins=True).to_csv(
                output_path / f'{prefix}_by_gender.csv', encoding='utf-8-sig')

        if 'age' in enriched.columns:
            age_data = enriched[enriched['age'] != 'Unknown']
            if not age_data.empty:
                age_ct = pd.crosstab(age_data['age'], age_data['particle'], margins=True)
                idx_order = [a for a in AGE_ORDER + ['All'] if a in age_ct.index]
                age_ct.reindex(idx_order).to_csv(
                    output_path / f'{prefix}_age_particle_crosstab.csv', encoding='utf-8-sig')

        print(f"CSV files saved to: {output_path}")

    def _export_excel(self, df_results: pd.DataFrame, enriched: pd.DataFrame,
                      output_path: Path, prefix: str):
        try:
            excel_file = output_path / f'{prefix}_results.xlsx'
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                df_results['particle'].value_counts().reset_index().rename(
                    columns={'particle': 'Particle (終助詞)', 'count': 'Count'}
                ).to_excel(writer, sheet_name='By Particle', index=False)

                if 'gender' in enriched.columns:
                    pd.crosstab(
                        enriched['gender'], enriched['particle'], margins=True
                    ).T.to_excel(writer, sheet_name='Gender x Particle')

                if 'age' in enriched.columns:
                    age_data = enriched[enriched['age'] != 'Unknown']
                    if not age_data.empty:
                        age_ct = pd.crosstab(age_data['age'], age_data['particle'], margins=True)
                        idx_order = [a for a in AGE_ORDER + ['All'] if a in age_ct.index]
                        age_ct.reindex(idx_order).to_excel(writer, sheet_name='Age x Particle')

                df_results['speaker'].value_counts().head(50).reset_index().rename(
                    columns={'speaker': 'Speaker', 'count': 'Count'}
                ).to_excel(writer, sheet_name='Top Speakers', index=False)

                df_results['conversation_id'].value_counts().reset_index().rename(
                    columns={'conversation_id': 'Conversation ID', 'count': 'Count'}
                ).to_excel(writer, sheet_name='By Conversation', index=False)

                for ws in writer.sheets.values():
                    ws.set_column(0, 10, 15)

            print(f"Excel file saved to: {excel_file}")
        except Exception as e:
            print(f"Warning: Could not create Excel file: {e}")

    # ----------------------------------------------------------
    # HTML export — single stacked file with one section per main speaker
    # ----------------------------------------------------------

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
        h1 {{ color: #333; border-bottom: 3px solid {accent}; padding: 18px 24px 12px;
              margin: 0; background: white; font-size: 20px; }}
        h2 {{ color: #444; margin: 24px 0 8px; font-size: 16px; }}
        h3 {{ color: #666; margin: 20px 0 4px; font-size: 14px; }}
        table {{ border-collapse: collapse; margin: 8px 0 20px; background: white;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.15); font-size: 13px; max-width: 100%;
                 overflow-x: auto; display: block; }}
        th, td {{ border: 1px solid #e0e0e0; padding: 7px 12px; text-align: right; white-space: nowrap; }}
        td {{ color: #000; }}
        th:first-child, td:first-child {{ text-align: left; font-weight: 600; color: #000;
                          background: #fafafa; position: sticky; left: 0; }}
        th {{ background: {accent}; color: white; font-weight: 600; }}
        tr:nth-child(even) td {{ background: #f9f9f9; }}
        tr:nth-child(even) td:first-child {{ background: #f3f3f3; }}
        tr:hover td {{ background: #eef4ff !important; }}
        
        .summary-box {{ display: flex; flex-wrap: wrap; gap: 16px; padding: 16px 24px;
                        background: white; margin-bottom: 4px; border-bottom: 1px solid #e0e0e0; }}
        .stat-item {{ text-align: center; min-width: 90px; }}
        .stat {{ font-size: 26px; color: {accent}; font-weight: 700; display: block; line-height: 1.1; }}
        .stat-label {{ font-size: 11px; color: #999; margin-top: 2px; }}
        .speaker-panel {{ padding: 20px 24px; display: block; }}
        .norm-note {{ font-size: 11px; color: #999; font-style: italic; margin: 2px 0 14px; }}
        .page-note {{ padding: 4px 24px 10px; color: #888; font-size: 12px;
                      background: white; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
{body}
</body>
</html>'''

    def _render_norm_table(
        self,
        df: pd.DataFrame,
        word_counts: Dict[Tuple[str, str], int],
        row_col: str,
        particles: List[str],
        row_order: Optional[List[str]] = None,
    ) -> str:
        """Render a normalized table: rows=row_col, cols=particles, values per 1000 words."""
        if df.empty:
            return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
        try:
            norm = self._normalize_by_word_count(df, word_counts, row_col, 'particle')
            cols = [p for p in particles if p in norm.columns]
            if not cols:
                return "<p style='color:#999;font-size:13px;'>データなし (No data)</p>\n"
            if row_order:
                norm = norm.reindex([r for r in row_order if r in norm.index])
            return norm[cols].to_html(float_format='%.2f', na_rep='-')
        except Exception as e:
            return f"<p style='color:red;font-size:13px;'>Error: {e}</p>\n"

    @staticmethod
    def _add_relative_age_col(df: pd.DataFrame) -> pd.DataFrame:
        """Add 'relative_age' column comparing partner age to main speaker age."""
        def _rel(row):
            try:
                diff = (int(row['partner_age'].split('-')[0])
                        - int(row['main_age'].split('-')[0]))
                if diff < -RELATIVE_AGE_THRESHOLD:
                    return '年下 (younger)'
                if diff > RELATIVE_AGE_THRESHOLD:
                    return '年上 (older)'
                return '同年代 (same age)'
            except Exception:
                return 'Unknown'
        df = df.copy()
        df['relative_age'] = df.apply(_rel, axis=1)
        return df

    def _build_speaker_tab_html(
        self,
        spk_df: pd.DataFrame,
        word_counts: Dict[Tuple[str, str], int],
        particles: List[str],
    ) -> str:
        """Build the HTML content for one main speaker's tab."""
        first       = spk_df.iloc[0]
        name        = first.get('main_speaker_name', '?')
        main_gender = first.get('main_gender', '?')
        main_age    = first.get('main_age', '?')
        n_conv      = spk_df['conversation_id'].nunique()
        n_sfp       = len(spk_df)

        pairs    = spk_df[['speaker', 'conversation_id']].drop_duplicates()
        n_words  = sum(
            word_counts.get((spk, conv), 0)
            for spk, conv in zip(pairs['speaker'], pairs['conversation_id'])
        )

        html = f'''<div class="summary-box">
    <div class="stat-item">
        <span class="stat">{n_conv}</span>
        <span class="stat-label">会話数<br>(conversations)</span>
    </div>
    <div class="stat-item">
        <span class="stat">{n_words:,}</span>
        <span class="stat-label">総語数<br>(words spoken)</span>
    </div>
    <div class="stat-item">
        <span class="stat">{n_sfp:,}</span>
        <span class="stat-label">SFP総数<br>(SFPs used)</span>
    </div>
    <div class="stat-item">
        <span class="stat">{main_gender}</span>
        <span class="stat-label">性別<br>(gender)</span>
    </div>
    <div class="stat-item">
        <span class="stat" style="font-size:18px;">{main_age}</span>
        <span class="stat-label">年齢<br>(age)</span>
    </div>
</div>
<div style="padding:8px 0 0;">
<p class="norm-note">
    すべての値 = {name} が 1,000 語あたりに使用する SFP 回数
    (All values = SFP occurrences per 1,000 words spoken by {name})
</p>
'''

        # Table 1: by partner gender
        html += "<h3>相手の性別別 SFP 使用率 (SFP rate by partner gender)</h3>\n"
        html += self._render_norm_table(spk_df, word_counts, 'partner_gender', particles)

        # Table 2: by partner age group
        valid_age_df = spk_df[spk_df['partner_age'] != 'Unknown'].copy()
        age_rows     = [a for a in AGE_ORDER if a in valid_age_df['partner_age'].unique()]
        html += "<h3>相手の年齢別 SFP 使用率 (SFP rate by partner age group)</h3>\n"
        html += self._render_norm_table(
            valid_age_df, word_counts, 'partner_age', particles, row_order=age_rows)

        # Table 3: by relative age
        rel_df = self._add_relative_age_col(valid_age_df)
        rel_df = rel_df[rel_df['relative_age'] != 'Unknown']
        html += (
            f"<h3>相手の相対年齢別 SFP 使用率 "
            f"(SFP rate by partner relative age, ±{RELATIVE_AGE_THRESHOLD}歳基準)</h3>\n"
        )
        html += self._render_norm_table(
            rel_df, word_counts, 'relative_age', particles,
            row_order=['年下 (younger)', '同年代 (same age)', '年上 (older)'])

        # Table 4: partner gender × relative age combined
        if not rel_df.empty:
            rel_df = rel_df.copy()
            rel_df['gender_rel'] = rel_df['partner_gender'] + ' / ' + rel_df['relative_age']
            combo_order = [
                f'{g} / {r}'
                for g in ['女性', '男性']
                for r in ['年下 (younger)', '同年代 (same age)', '年上 (older)']
            ]
            html += (
                "<h3>相手の性別 × 相対年齢別 SFP 使用率 "
                "(SFP rate by partner gender × relative age)</h3>\n"
            )
            html += self._render_norm_table(
                rel_df, word_counts, 'gender_rel', particles, row_order=combo_order)

        html += "</div>\n"
        return html

    def _export_html_tabbed(
        self,
        df_results: pd.DataFrame,
        word_counts: Dict[Tuple[str, str], int],
        output_path: Path,
        prefix: str,
    ):
        """Write a single HTML file with one visible section per main speaker."""
        try:
            partner_data = self.analyze_sfp_by_partner(df_results)
            if partner_data.empty:
                print("Warning: No partner data — skipping HTML export.")
                return

            # Keep only main speakers with enough conversations
            grouped = {
                sid: grp
                for sid, grp in partner_data.groupby('main_speaker_id')
                if grp['conversation_id'].nunique() >= MIN_SPEAKER_CONVERSATIONS
            }
            if not grouped:
                print(
                    f"Warning: No main speakers with ≥{MIN_SPEAKER_CONVERSATIONS} "
                    f"conversations — skipping HTML export."
                )
                return

            # Sort alphabetically by speaker name for a stable tab order
            sorted_speakers = sorted(
                grouped.items(),
                key=lambda kv: kv[1].iloc[0].get('main_speaker_name', kv[0])
            )

            speaker_content_html = ''
            for sid, spk_df in sorted_speakers:
                name   = spk_df.iloc[0].get('main_speaker_name', sid)
                age    = spk_df.iloc[0].get('main_age', '?')
                gender = spk_df.iloc[0].get('main_gender', '?')
                tab_id = f'spk_{sid}'

                content = self._build_speaker_tab_html(spk_df, word_counts, REPORT_PARTICLES)
                speaker_content_html += (
                    f'<div id="{tab_id}" class="speaker-panel">\n'
                    f'<h2>{name} — SFP使用パターン（相手属性別）</h2>\n'
                    f'{content}\n</div>\n'
                )

            n_speakers = len(sorted_speakers)
            body = (
                f'<h1>SFP Analysis by Main Speaker — 話者別終助詞分析</h1>\n'
                f'<p class="page-note">'
                f'{n_speakers} 名の主話者 · 値 = 1,000 語あたり SFP 使用数</p>\n'
                f'{speaker_content_html}'
            )

            out_file = output_path / f'{prefix}_main_speaker_analysis.html'
            out_file.write_text(
                self._html_page('SFP Analysis by Main Speaker', '#4CAF50', body),
                encoding='utf-8',
            )
            print(f"HTML file saved to: {out_file}")
        except Exception as e:
            import traceback
            print(f"Warning: Could not create HTML file: {e}")
            traceback.print_exc()
            
    # ----------------------------------------------------------
    # Analysis methods
    # ----------------------------------------------------------

    @staticmethod
    def _get_age_category(age_str: str) -> str:
        if pd.isna(age_str) or age_str == 'Unknown':
            return 'Unknown'
        try:
            start = int(age_str.split('-')[0])
            if start < 20: return 'young (0-19)'
            if start < 40: return 'adult (20-39)'
            if start < 60: return 'middle (40-59)'
            return 'senior (60+)'
        except Exception:
            return 'Unknown'

    def analyze_sfp_by_partner(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Enrich main-speaker SFP rows with conversation partner characteristics."""
        if df_results.empty or self.speaker_data.empty:
            return pd.DataFrame()

        scr_path = self.cejc_root / 'Speaker_Conversation_Relation.csv'
        if not scr_path.exists():
            print("Warning: Speaker_Conversation_Relation.csv not found")
            return pd.DataFrame()
        scr = pd.read_csv(scr_path, encoding=self.encoding)

        speaker_info = {}
        for _, row in self.speaker_data.iterrows():
            sid = row.get(COL_SPEAKER_ID, '')
            if pd.notna(sid):
                speaker_info[sid] = {
                    'name':   row.get(COL_SPEAKER_NAME, ''),
                    'gender': row.get(COL_GENDER) if pd.notna(row.get(COL_GENDER)) else None,
                    'age':    row.get(COL_AGE)    if pd.notna(row.get(COL_AGE))    else None,
                }

        # Main speakers have no underscore in their speaker ID
        main_speaker_ids = {sid for sid in speaker_info if '_' not in sid}

        conv_partners = {}
        for conv_id in df_results['conversation_id'].unique():
            participants = scr[scr[COL_CONV_ID] == conv_id][COL_SPEAKER_ID].tolist()
            main_spk = next((p for p in participants if p in main_speaker_ids), None)
            partner  = next((p for p in participants
                             if p != main_spk and p in speaker_info), None)
            if not (main_spk and partner):
                continue
            m  = speaker_info[main_spk]
            pt = speaker_info[partner]
            if not all([m.get('gender'), m.get('age'), pt.get('gender'), pt.get('age')]):
                continue
            conv_partners[conv_id] = {
                'main_speaker_id':      main_spk,
                'main_speaker_name':    m['name'],
                'main_gender':          m['gender'],
                'main_age':             m['age'],
                'partner_gender':       pt['gender'],
                'partner_age':          pt['age'],
                'partner_age_category': self._get_age_category(pt['age']),
            }

        results = []
        for _, row in df_results.iterrows():
            conv_id = row['conversation_id']
            if conv_id not in conv_partners:
                continue
            info = conv_partners[conv_id]
            if self._extract_speaker_name(row['speaker']) == info['main_speaker_name']:
                results.append({**row.to_dict(), **info})

        return pd.DataFrame(results)

    def analyze_by_conversation_type(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """SFP counts by conversation type (first character of conversation ID)."""
        if df_results.empty:
            return pd.DataFrame()
        df = df_results.copy()
        df['conv_type'] = df['conversation_id'].str[0]
        return df.groupby(['conv_type', 'particle']).size().unstack(fill_value=0)

    def analyze_sfp_cooccurrence(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Count compound vs single particle occurrences."""
        if df_results.empty:
            return pd.DataFrame()
        compound_labels = {label for _, _, label in COMPOUND_RULES}
        df = df_results.copy()
        df['is_compound'] = df['particle'].isin(compound_labels)
        return df.groupby(['particle', 'is_compound']).size().reset_index(name='count')


def plot_adaptation_trends(counter_instance: SFPCounter, df_results: pd.DataFrame, word_counts: dict, output_dir: str):
    """
    Calculates SFP variance and plots the adaptation distribution 
    across different opponent demographic groups.
    Organized by Main Speaker Gender (Male first, then Female).
    """
    if df_results.empty:
        print("No data available to plot adaptation trends.")
        return

    # 1. Enrich data with partner info and relative age
    partner_df = counter_instance.analyze_sfp_by_partner(df_results)
    if partner_df.empty:
        print("No partner data available for plotting.")
        return

    partner_df = counter_instance._add_relative_age_col(partner_df)
    partner_df = partner_df[partner_df['relative_age'] != 'Unknown']

    # 2. Calculate SFP rate PER CONVERSATION for distributions
    # Added 'main_gender' and 'main_age' to the groupby so we can use them in titles
    conv_stats = partner_df.groupby(
        ['main_speaker_name', 'main_gender', 'main_age', 'speaker', 'conversation_id', 'partner_gender', 'relative_age']
    ).size().reset_index(name='sfp_count')

    # Apply word counts to get normalized rates
    conv_stats['word_count'] = conv_stats.apply(
        lambda row: word_counts.get((row['speaker'], row['conversation_id']), 1), axis=1
    )
    conv_stats['word_count'] = conv_stats['word_count'].replace(0, 1) # Prevent div zero
    conv_stats['sfp_rate'] = (conv_stats['sfp_count'] / conv_stats['word_count']) * 1000

    # Filter out speakers with too few conversations
    valid_speakers = conv_stats['main_speaker_name'].value_counts()
    valid_speakers = valid_speakers[valid_speakers >= MIN_SPEAKER_CONVERSATIONS].index
    conv_stats = conv_stats[conv_stats['main_speaker_name'].isin(valid_speakers)]

    # 3. Create Custom Display Names (Name + Age + Gender)
    conv_stats['display_name'] = conv_stats.apply(
        lambda row: f"{row['main_speaker_name']} ({row['main_age']}, {row['main_gender']})", axis=1
    )

    # --- NEW: Create Overall Aggregate Data ---
    # Pool all male data into a single group
    male_data = conv_stats[conv_stats['main_gender'].isin(['男性', 'M', 'Male'])].copy()
    male_data['display_name'] = 'Overall Male (男性 全体)'
    male_data['main_speaker_name'] = '0_Overall_Male' # Prefix forces it to the top alphabetically later
    
    # Pool all female data into a single group
    female_data = conv_stats[conv_stats['main_gender'].isin(['女性', 'F', 'Female'])].copy()
    female_data['display_name'] = 'Overall Female (女性 全体)'
    female_data['main_speaker_name'] = '1_Overall_Female'

    # Append these aggregate groups back into the main dataframe
    conv_stats = pd.concat([male_data, female_data, conv_stats], ignore_index=True)
    # ----------------------------------------

    # 4. Sort Speakers: Overall first, Male first, then Female
    def gender_sort_key(gender_str, name_str):
        if '0_Overall_Male' in name_str: return -2     # Always 1st
        if '1_Overall_Female' in name_str: return -1   # Always 2nd
        
        g = str(gender_str).lower()
        if g in ['男性', 'm', 'male']: return 0        # 3rd: Individual Males
        if g in ['女性', 'f', 'female']: return 1      # 4th: Individual Females
        return 2                                       # Fallback

    unique_speakers = conv_stats[['display_name', 'main_gender', 'main_speaker_name']].drop_duplicates()
    
    # Apply the sorting rank
    unique_speakers['sort_rank'] = unique_speakers.apply(
        lambda row: gender_sort_key(row['main_gender'], row['main_speaker_name']), axis=1
    )
    
    # Sort primarily by rank, then alphabetically by name to keep it neat
    unique_speakers = unique_speakers.sort_values(by=['sort_rank', 'main_speaker_name'])
    
    # Extract the final ordered list for the FacetGrid
    speaker_order = unique_speakers['display_name'].tolist()

    # 5. Calculate and export statistical variance (分散)
    variance_stats = conv_stats.groupby('display_name')['sfp_rate'].var().fillna(0).reset_index()
    variance_stats.rename(columns={'sfp_rate': 'sfp_variance'}, inplace=True)
    variance_stats.sort_values(by='sfp_variance', ascending=False, inplace=True)

    variance_csv_path = Path(output_dir) / 'sfp_adaptation_variance.csv'
    variance_stats.to_csv(variance_csv_path, index=False, encoding='utf-8-sig')

    # 6. Plotting setup
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    age_order = ['年下 (younger)', '同年代 (same age)', '年上 (older)']
    
    # Custom color mapping for the BOXES (Female = Red, Male = Blue)
    box_palette = {
        '女性': '#e74c3c', 'F': '#e74c3c', 'Female': '#e74c3c',
        '男性': '#3498db', 'M': '#3498db', 'Male': '#3498db'
    }

    # Custom color mapping for the DOTS (Female = Green, Male = Orange)
    dot_palette = {
        '女性': '#2ecc71', 'F': '#2ecc71', 'Female': '#2ecc71',
        '男性': '#f39c12', 'M': '#f39c12', 'Male': '#f39c12'
    }

    # 7. Generate FacetGrid with Boxplots
    # Define hue order so colors never swap between panels
    hue_order = ['男性', '女性'] 

    g = sns.catplot(
        data=conv_stats,
        x='relative_age',
        y='sfp_rate',
        hue='partner_gender',
        hue_order=hue_order,  # <-- ADD THIS LINE
        col='display_name',
        col_wrap=3,
        col_order=speaker_order, 
        kind='box',
        order=age_order,
        height=4,
        aspect=1.2,
        sharey=True,
        palette=box_palette
    )

    # Overlay actual data points (Stripplot)
    g.map_dataframe(
        sns.stripplot,
        x='relative_age',
        y='sfp_rate',
        hue='partner_gender',
        hue_order=hue_order,  # <-- ADD THIS LINE
        order=age_order,
        dodge=True,
        alpha=0.8,
        palette=dot_palette 
    )

    g.set_axis_labels("相手の相対年齢 (Opponent Relative Age)", "SFP Rate (per 1,000 words)")
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle('SFP Usage Adaptation by Main Speaker (Opponent Age & Gender)', fontsize=16)

    plot_path = Path(output_dir) / 'sfp_adaptation_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_adaptation_trends_by_particle(counter_instance, df_results: pd.DataFrame, word_counts: dict, output_dir: str, target_particles=['ね', 'よ', 'な', 'わ', 'よね']):
    """
    Calculates SFP variance by specific particle type and plots the adaptation 
    distribution. Generates a separate image for each target particle.
    """
    if df_results.empty:
        print("No data available to plot adaptation trends.")
        return

    # 1. Enrich data with partner info and relative age
    partner_df = counter_instance.analyze_sfp_by_partner(df_results)
    if partner_df.empty:
        print("No partner data available for plotting.")
        return

    partner_df = counter_instance._add_relative_age_col(partner_df)
    partner_df = partner_df[partner_df['relative_age'] != 'Unknown']

    # 2. Group by EVERYTHING including 'particle'
    conv_stats = partner_df.groupby(
        ['main_speaker_name', 'main_gender', 'main_age', 'speaker', 'conversation_id', 'partner_gender', 'relative_age', 'particle']
    ).size().reset_index(name='sfp_count')

    # Apply word counts to get normalized rates
    conv_stats['word_count'] = conv_stats.apply(
        lambda row: word_counts.get((row['speaker'], row['conversation_id']), 1), axis=1
    )
    conv_stats['word_count'] = conv_stats['word_count'].replace(0, 1) # Prevent div zero
    conv_stats['sfp_rate'] = (conv_stats['sfp_count'] / conv_stats['word_count']) * 1000

    # Filter out speakers with too few conversations
    valid_speakers = conv_stats['main_speaker_name'].value_counts()
    valid_speakers = valid_speakers[valid_speakers >= MIN_SPEAKER_CONVERSATIONS].index
    conv_stats = conv_stats[conv_stats['main_speaker_name'].isin(valid_speakers)]

    # 3. Create Custom Display Names (Name + Age + Gender)
    conv_stats['display_name'] = conv_stats.apply(
        lambda row: f"{row['main_speaker_name']} ({row['main_age']}, {row['main_gender']})", axis=1
    )

    # --- NEW: Create Overall Aggregate Data ---
    # Pool all male data into a single group
    male_data = conv_stats[conv_stats['main_gender'].isin(['男性', 'M', 'Male'])].copy()
    male_data['display_name'] = 'Overall Male (男性 全体)'
    male_data['main_speaker_name'] = '0_Overall_Male' # Prefix forces it to the top alphabetically later
    
    # Pool all female data into a single group
    female_data = conv_stats[conv_stats['main_gender'].isin(['女性', 'F', 'Female'])].copy()
    female_data['display_name'] = 'Overall Female (女性 全体)'
    female_data['main_speaker_name'] = '1_Overall_Female'

    # Append these aggregate groups back into the main dataframe
    conv_stats = pd.concat([male_data, female_data, conv_stats], ignore_index=True)
    # ----------------------------------------

    # 4. Sort Speakers: Overall first, Male first, then Female
    def gender_sort_key(gender_str, name_str):
        if '0_Overall_Male' in name_str: return -2     # Always 1st
        if '1_Overall_Female' in name_str: return -1   # Always 2nd
        
        g = str(gender_str).lower()
        if g in ['男性', 'm', 'male']: return 0        # 3rd: Individual Males
        if g in ['女性', 'f', 'female']: return 1      # 4th: Individual Females
        return 2                                       # Fallback

    unique_speakers = conv_stats[['display_name', 'main_gender', 'main_speaker_name']].drop_duplicates()
    
    # Apply the sorting rank
    unique_speakers['sort_rank'] = unique_speakers.apply(
        lambda row: gender_sort_key(row['main_gender'], row['main_speaker_name']), axis=1
    )
    
    # Sort primarily by rank, then alphabetically by name to keep it neat
    unique_speakers = unique_speakers.sort_values(by=['sort_rank', 'main_speaker_name'])
    
    # Extract the final ordered list for the FacetGrid
    speaker_order = unique_speakers['display_name'].tolist()

    # 5. Plotting setup
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    age_order = ['年下 (younger)', '同年代 (same age)', '年上 (older)']
    
    box_palette = {
        '女性': '#e74c3c', 'F': '#e74c3c', 'Female': '#e74c3c',
        '男性': '#3498db', 'M': '#3498db', 'Male': '#3498db'
    }

    dot_palette = {
        '女性': '#2ecc71', 'F': '#2ecc71', 'Female': '#2ecc71',
        '男性': '#f39c12', 'M': '#f39c12', 'Male': '#f39c12'
    }

    # 6. Generate a separate plot for EACH target particle
    for particle in target_particles:
        part_stats = conv_stats[conv_stats['particle'] == particle]
        
        # Skip if nobody used this particle enough to plot
        if part_stats.empty:
            continue

        g = sns.catplot(
            data=part_stats,
            x='relative_age',
            y='sfp_rate',
            hue='partner_gender',
            col='display_name',
            col_wrap=3,
            col_order=speaker_order, 
            kind='box',
            order=age_order,
            height=4,
            aspect=1.2,
            sharey=True, # Set to False so the Y-axis scales nicely for less common particles
            palette=box_palette
        )

        g.map_dataframe(
            sns.stripplot,
            x='relative_age',
            y='sfp_rate',
            hue='partner_gender',
            order=age_order,
            dodge=True,
            alpha=0.8,
            palette=dot_palette
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

    # Map relative_age labels to short display labels per opponent gender
    rel_age_map = {
        '年下 (younger)': 'Younger',
        '同年代 (same age)': 'Same Age',
        '年上 (older)': 'Older',
    }
    filtered_df['rel_age_short'] = filtered_df['relative_age'].map(rel_age_map)

    male_gender_vals   = ['男性', 'M', 'Male']
    female_gender_vals = ['女性', 'F', 'Female']

    # Build the 6 opponent buckets: M/Younger, M/Same Age, M/Older, F/Younger, F/Same Age, F/Older
    opponent_labels = ['M/Younger', 'M/Same Age', 'M/Older', 'F/Younger', 'F/Same Age', 'F/Older']

    def compute_pct(df, main_gender_vals, partner_gender_vals, rel_ages):
        """Compute percentage share of each top-5 SFP for given speaker/opponent combo."""
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

    # Build percentage data for each speaker gender across the 6 opponent buckets
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

        # Invisible NaN plot to anchor the full x-axis before splitting
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
    
def main():
    parser = argparse.ArgumentParser(description='CEJC Sentence Final Particle Counter')
    parser.add_argument(
        '--csv-output',
        action='store_true',
        help='Generate CSV output files in addition to Excel and HTML outputs.',
    )
    parser.add_argument(
        '--sfp-mode',
        choices=['old', 'new'],
        default=SFP_MODE,
        help=(
            '"old" = original SFP set (default). '
            '"new" = gendered taxonomy from the 3-column table (algorithm-detectable subset only; '
            'items requiring LLM/prosody context are excluded).'
        ),
    )
    args = parser.parse_args()

    if args.sfp_mode == 'new':
        active_sfps     = NEW_TARGET_SFPS
        active_compounds = NEW_COMPOUND_RULES
        active_report   = NEW_REPORT_PARTICLES
        mode_label      = 'new (gendered taxonomy)'
    else:
        active_sfps     = TARGET_SFPS
        active_compounds = COMPOUND_RULES
        active_report   = REPORT_PARTICLES
        mode_label      = 'old (original)'

    print("=" * 60)
    print(f"CEJC Sentence Final Particle Counter — mode: {mode_label}")
    print("=" * 60)

    counter = SFPCounter()  # uses CEJC_ROOT and FILE_ENCODING constants

    # Override module-level rule/report lists so all downstream functions
    # (count_all, export_results, plots) pick up the selected mode's sets.
    import sys
    this_module = sys.modules[__name__]
    this_module.COMPOUND_RULES   = active_compounds   # type: ignore[attr-defined]
    this_module.REPORT_PARTICLES = active_report       # type: ignore[attr-defined]

    df_results, word_counts = counter.count_all(target_sfps=active_sfps)
    summary = counter.get_summary(df_results)

    print(f"\nTotal SFP occurrences: {summary['total_count']}")
    print("\nBy particle:")
    for particle, count in sorted(summary['by_particle'].items(),
                                  key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {particle}: {count}")

    counter.export_results(
        df_results,
        OUTPUT_DIR,
        word_counts=word_counts,
        csv_output=args.csv_output,
    )

    print("\nGenerating adaptation distribution graphs...")
    plot_adaptation_trends(counter, df_results, word_counts, OUTPUT_DIR)

    if args.sfp_mode == 'new':
        target_particles_to_plot = ['ね', 'よ', 'わ', 'よね', 'な', 'ぞ', 'ぜ']
    else:
        target_particles_to_plot = ['ね', 'よ', 'な', 'よね', 'の']
    plot_adaptation_trends_by_particle(counter, df_results, word_counts, OUTPUT_DIR, target_particles_to_plot)

    plot_sfp_composition_by_opponent(counter, df_results, OUTPUT_DIR)

    print("\nDone!")


if __name__ == '__main__':
    main()
