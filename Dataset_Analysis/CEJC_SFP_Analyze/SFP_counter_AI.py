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
import os
import re
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
AI_OUTPUT_DIR = '/home/ryuu/Ryu/Dataset_Analysis/CEJC_SFP_Analyze/output/ai_pilot'
FILE_ENCODING = 'cp932'   # Shift-JIS; change to 'utf-8' if corpus is re-encoded

# Full gendered SFP taxonomy (grammar-pattern entries included, e.g. 名詞＋よ, て形＋ね),
# used as the closed label set for Gemini classification in --sfp-mode ai.
SFP_TAXONOMY_PATH = (
    '/home/ryuu/Ryu/System_implement/LLM_Response/SFP_Bucket_by_gender.json'
)
DEFAULT_AI_CONVERSATION = 'C001_002'  # single-conversation pilot scope
DEFAULT_AI_FOLDER = 'C001'            # top-level folder scope for --sfp-mode ai

# Per-conversation classification results are cached here so an interrupted --sweep-all
# (or --folder) run can resume without re-classifying conversations already done.
AI_CACHE_DIR = '/home/ryuu/Ryu/Dataset_Analysis/CEJC_SFP_Analyze/output/ai_cache'
AI_CLASSIFY_MAX_WORKERS = 8  # concurrent Gemini calls per conversation

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


# ==============================================================
# GEMINI CLIENT (for --sfp-mode ai) — same pattern as
# System_implement/LLM_Response/Response_All_AI.py get_client()
# ==============================================================
_genai_client = None
_genai_client_lock = threading.Lock()


def get_genai_client():
    global _genai_client
    if not _genai_client:
        with _genai_client_lock:
            if not _genai_client:
                from google import genai
                if "GENAI_API_KEY" not in os.environ:
                    raise RuntimeError("GENAI_API_KEY is not set")
                _genai_client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
    return _genai_client


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

    # ----------------------------------------------------------
    # Gemini-based classification (--sfp-mode ai) — pilot, single conversation
    # ----------------------------------------------------------

    @staticmethod
    def _load_sfp_taxonomy() -> Dict[str, List[str]]:
        with open(SFP_TAXONOMY_PATH, 'r', encoding='utf-8') as handle:
            return json.load(handle)

    def _find_conversation_file(self, conversation_id: str) -> Optional[Path]:
        matches = list(self.cejc_root.glob(f'**/{conversation_id}-morphSUW.csv'))
        return matches[0] if matches else None

    def _reconstruct_utterances(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Group tokens into utterances by (speaker, utt_start, utt_end), preserving order."""
        if df.empty:
            return []
        utterances = []
        for (speaker, start, end), group in df.groupby(
            [COL_SPEAKER, COL_UTT_START, COL_UTT_END], sort=False
        ):
            utterances.append({
                'speaker':  speaker,
                'start':    start,
                'end':      end,
                'text':     ''.join(str(s) for s in group[COL_SURFACE]),
                'tokens':   group,
            })
        return utterances

    @staticmethod
    def _build_ai_classification_prompt(
        taxonomy: Dict[str, List[str]], utterance_text: str, marked_particle: str,
        preceding_context: List[str],
    ) -> str:
        taxonomy_lines = []
        for bucket, entries in taxonomy.items():
            taxonomy_lines.append(f"[{bucket}]")
            for entry in entries:
                taxonomy_lines.append(f"  - {entry}")
        taxonomy_block = "\n".join(taxonomy_lines)

        context_block = (
            "\n".join(preceding_context) if preceding_context
            else "(no preceding context)"
        )

        return (
            "You are classifying a Japanese sentence-final particle (終助詞) against a "
            "gendered SFP taxonomy. Some taxonomy entries are grammatical patterns "
            "(e.g. 名詞＋よ, 形容動詞＋の, 動詞の命令形＋よ, て形＋ね) that depend on the "
            "part-of-speech or conjugation form of the word immediately before the particle, "
            "not just the particle's surface form — use the full utterance to judge this.\n\n"
            f"Taxonomy (bucket: entries):\n{taxonomy_block}\n\n"
            f"Preceding conversation context:\n{context_block}\n\n"
            f"Utterance to classify (target ending marked with 【】): {utterance_text}\n"
            f"Marked particle/ending: {marked_particle}\n\n"
            "Return ONLY a JSON object with keys: "
            "\"bucket\" (one of feminine, neutral, masculine, or none), "
            "\"matched_entry\" (the exact taxonomy entry string that best matches, or null if none fit)."
        )

    def _classify_sfp_with_ai(
        self, taxonomy: Dict[str, List[str]], utterance_text: str, marked_particle: str,
        preceding_context: List[str],
    ) -> Dict[str, Any]:
        prompt = self._build_ai_classification_prompt(
            taxonomy, utterance_text, marked_particle, preceding_context)
        response = get_genai_client().models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        ).text.strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {'bucket': None, 'matched_entry': None}

    @staticmethod
    def _conversation_cache_path(conversation_id: str) -> Path:
        return Path(AI_CACHE_DIR) / f'{conversation_id}.csv'

    def classify_conversation_with_ai(
        self, conversation_id: str = DEFAULT_AI_CONVERSATION, context_window: int = 3,
        max_workers: int = AI_CLASSIFY_MAX_WORKERS, use_cache: bool = True,
    ) -> pd.DataFrame:
        """Classify every utterance-final SFP candidate in one conversation via Gemini,
        against the full gendered taxonomy. Classifications run concurrently (max_workers
        at a time). Results are cached to AI_CACHE_DIR so a resumed sweep can skip
        conversations already classified."""
        cache_path = self._conversation_cache_path(conversation_id)
        if use_cache and cache_path.exists():
            print(f"Using cached classification for {conversation_id} ({cache_path.name})")
            return pd.read_csv(cache_path)

        filepath = self._find_conversation_file(conversation_id)
        if filepath is None:
            print(f"Warning: no morphSUW file found for conversation {conversation_id}")
            return pd.DataFrame()

        df = self._load_morph_file(filepath)
        if df.empty:
            return pd.DataFrame()

        taxonomy = self._load_sfp_taxonomy()
        utterances = self._reconstruct_utterances(df)
        name_to_gender, name_to_age = self._build_speaker_lookup()

        # Build the candidate list first so context (preceding utterances) is computed
        # from the sequential transcript, then classify candidates concurrently.
        candidates = []
        for i, utt in enumerate(utterances):
            tokens = utt['tokens']
            if tokens.empty or tokens.iloc[-1][COL_POS] != POS_SENTENCE_FINAL:
                continue
            marked_particle = tokens.iloc[-1][COL_SURFACE]
            marked_text = utt['text'][:-len(str(marked_particle))] + f"【{marked_particle}】"
            preceding_context = [
                f"{utterances[j]['speaker']}: {utterances[j]['text']}"
                for j in range(max(0, i - context_window), i)
            ]
            candidates.append((i, utt, marked_particle, marked_text, preceding_context))

        results = [None] * len(candidates)
        progress_lock = threading.Lock()
        done_count = 0

        def classify_one(idx):
            _, utt, marked_particle, marked_text, preceding_context = candidates[idx]
            result = self._classify_sfp_with_ai(
                taxonomy, marked_text, marked_particle, preceding_context)
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(classify_one, idx): idx for idx in range(len(candidates))}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                with progress_lock:
                    done_count += 1
                    _, utt, marked_particle, marked_text, _ = candidates[idx]
                    print(f"[{done_count}/{len(candidates)}] {utt['speaker']}: {marked_text} -> "
                          f"{result.get('bucket')} / {result.get('matched_entry')}")

        rows = []
        for (i, utt, marked_particle, marked_text, _), result in zip(candidates, results):
            speaker_name = self._extract_speaker_name(utt['speaker'])
            rows.append({
                'conversation_id': conversation_id,
                'speaker':         utt['speaker'],
                'speaker_name':    speaker_name,
                'speaker_gender':  name_to_gender.get(speaker_name),
                'speaker_age':     name_to_age.get(speaker_name),
                'utterance_text':  utt['text'],
                'marked_particle': marked_particle,
                'ai_bucket':       result.get('bucket'),
                'ai_matched_entry': result.get('matched_entry'),
                'start_time':      utt['start'],
                'end_time':        utt['end'],
            })

        df_result = pd.DataFrame(rows)

        if use_cache and not df_result.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_result.to_csv(cache_path, index=False, encoding='utf-8-sig')

        return df_result

    def find_two_speaker_conversations_in_folder(self, folder_id: str) -> List[str]:
        """List conversation IDs under a top-level folder (e.g. 'C001', 'K001') that have
        exactly 2 speakers. Conversation IDs come from the morphSUW filenames themselves
        (not directory names), since some conversations are split into suffixed sub-IDs
        that share one directory (e.g. dir K001_003 holds K001_003a and K001_003b)."""
        morph_files = (self.cejc_root / folder_id).glob('**/*-morphSUW.csv')
        conv_ids = sorted({f.name.replace('-morphSUW.csv', '') for f in morph_files})

        two_speaker_ids = set(self._get_two_speaker_conversations())
        if two_speaker_ids:
            return [c for c in conv_ids if c in two_speaker_ids]
        return [
            c for c in conv_ids
            if self._find_conversation_file(c) is not None
            and self._count_speakers(self._load_morph_file(self._find_conversation_file(c))) == 2
        ]

    def _get_conversation_partner_info(self, conversation_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """For each conversation, find the main speaker (no underscore in 話者ID) and their
        partner, returning gender/age for both. Reuses the join logic from analyze_sfp_by_partner."""
        scr_path = self.cejc_root / 'Speaker_Conversation_Relation.csv'
        if not scr_path.exists() or self.speaker_data.empty:
            return {}
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

        main_speaker_ids = {sid for sid in speaker_info if '_' not in sid}

        conv_partners = {}
        for conv_id in conversation_ids:
            participants = scr[scr[COL_CONV_ID] == conv_id][COL_SPEAKER_ID].tolist()
            main_spk = next((p for p in participants if p in main_speaker_ids), None)
            partner  = next((p for p in participants
                             if p != main_spk and p in speaker_info), None)
            if not (main_spk and partner):
                continue
            m, pt = speaker_info[main_spk], speaker_info[partner]
            if not all([m.get('gender'), m.get('age'), pt.get('gender'), pt.get('age')]):
                continue
            conv_partners[conv_id] = {
                'main_speaker_id':   main_spk,
                'main_speaker_name': m['name'],
                'main_gender':       m['gender'],
                'main_age':          m['age'],
                'partner_name':      pt['name'],
                'partner_gender':    pt['gender'],
                'partner_age':       pt['age'],
            }
        return conv_partners

    def classify_folder_with_ai(self, folder_id: str, context_window: int = 3) -> pd.DataFrame:
        """Classify SFPs across every 1-on-1 conversation under a top-level folder (e.g. C001),
        tagging each row with main-speaker/partner conversation metadata."""
        conv_ids = self.find_two_speaker_conversations_in_folder(folder_id)
        if not conv_ids:
            print(f"Warning: no 1-on-1 conversations found under {folder_id}")
            return pd.DataFrame()

        partner_info = self._get_conversation_partner_info(conv_ids)

        all_rows = []
        for conv_id in conv_ids:
            if conv_id not in partner_info:
                print(f"Skipping {conv_id}: missing speaker/partner metadata")
                continue
            print(f"\n=== Classifying {conv_id} ===")
            df_conv = self.classify_conversation_with_ai(conv_id, context_window=context_window)
            if df_conv.empty:
                continue
            info = partner_info[conv_id]
            df_conv = df_conv[df_conv['speaker_name'] == info['main_speaker_name']].copy()
            for key, val in info.items():
                df_conv[key] = val
            all_rows.append(df_conv)

        if not all_rows:
            return pd.DataFrame()
        return pd.concat(all_rows, ignore_index=True)

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

    plot_path = Path(output_dir) / 'sfp_adaptation_distribution.svg'
    plt.savefig(plot_path, bbox_inches='tight')
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

        plot_path = Path(output_dir) / f'sfp_adaptation_distribution_{particle}.svg'
        plt.savefig(plot_path, bbox_inches='tight')
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
    plot_path = Path(output_dir) / 'sfp_top5_composition_grouped.svg'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved small-multiples composition chart: {plot_path.name}")


def plot_ai_bucket_distribution(df_ai: pd.DataFrame, conversation_id: str, output_dir: str):
    """Bar chart of masculine/neutral/feminine SFP share (%) out of all classified rows."""
    if df_ai.empty:
        return

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    bucket_order = ['masculine', 'neutral', 'feminine']
    counts = df_ai['ai_bucket'].value_counts()
    total = len(df_ai)
    pct = pd.Series({b: counts.get(b, 0) / total * 100 for b in bucket_order})

    colors = {'masculine': '#3498db', 'neutral': '#95a5a6', 'feminine': '#e74c3c'}

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(pct.index, pct.values, color=[colors[b] for b in bucket_order])
    for bar, b in zip(bars, bucket_order):
        n = counts.get(b, 0)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{pct[b]:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Percentage of SFP lines (%)')
    ax.set_ylim(0, max(pct.values) * 1.25 if max(pct.values) > 0 else 100)
    ax.set_title(f'SFP Gender Bucket Distribution — {conversation_id} (n={total})')

    plot_path = Path(output_dir) / f'sfp_ai_bucket_distribution_{conversation_id}.svg'
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved AI bucket distribution chart: {plot_path.name}")


RELATIVE_AGE_BIN_WIDTH = 5  # years per bin, e.g. "0-4 older", "5-9 younger"


def _relative_age_bin(partner_age: str, main_age: str) -> Optional[str]:
    """Bucket (partner_age - main_age) into 5-year-wide bins, e.g. '5-9 older', '0-4 younger'."""
    try:
        partner_start = int(str(partner_age).split('-')[0])
        main_start = int(str(main_age).split('-')[0])
    except (ValueError, IndexError):
        return None
    diff = partner_start - main_start
    direction = 'older' if diff >= 0 else 'younger'
    bin_start = (abs(diff) // RELATIVE_AGE_BIN_WIDTH) * RELATIVE_AGE_BIN_WIDTH
    bin_end = bin_start + RELATIVE_AGE_BIN_WIDTH - 1
    if bin_start == 0 and diff == 0:
        return 'same age'
    return f'{bin_start}-{bin_end} {direction}'


def _relative_age_bin_sort_key(label: str):
    if label == 'same age':
        return (0, 0)
    bin_start = int(label.split('-')[0])
    direction_rank = -1 if 'younger' in label else 1
    return (direction_rank, bin_start * direction_rank)


def plot_ai_relative_age_by_gender(df_ai: pd.DataFrame, folder_id: str, output_dir: str):
    """Two bar charts (male opponents, female opponents) showing the main speaker's SFP
    bucket % across 5-year partner-relative-age bins."""
    if df_ai.empty or 'partner_age' not in df_ai.columns or 'main_age' not in df_ai.columns:
        return

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    df = df_ai.copy()
    df['relative_age_bin'] = df.apply(
        lambda r: _relative_age_bin(r['partner_age'], r['main_age']), axis=1)
    df = df[df['relative_age_bin'].notna() & df['ai_bucket'].notna()]

    bucket_order = ['masculine', 'neutral', 'feminine']
    colors = {'masculine': '#3498db', 'neutral': '#95a5a6', 'feminine': '#e74c3c'}

    gender_groups = [('男性', 'male_opponents'), ('女性', 'female_opponents')]

    for partner_gender, file_tag in gender_groups:
        sub = df[df['partner_gender'] == partner_gender]
        if sub.empty:
            continue

        bin_order = sorted(sub['relative_age_bin'].unique(), key=_relative_age_bin_sort_key)
        pct_table = (
            sub.groupby('relative_age_bin')['ai_bucket']
            .value_counts(normalize=True)
            .unstack(fill_value=0.0) * 100
        ).reindex(index=bin_order, columns=bucket_order, fill_value=0.0)
        n_per_bin = sub.groupby('relative_age_bin').size().reindex(bin_order)

        x = range(len(bin_order))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(7, len(bin_order) * 1.5), 5))
        for i, bucket in enumerate(bucket_order):
            offset = (i - 1) * width
            bars = ax.bar([xi + offset for xi in x], pct_table[bucket], width,
                           label=bucket, color=colors[bucket])
            for bar, pct in zip(bars, pct_table[bucket]):
                if pct > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                             f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f'{b}\n(n={n_per_bin[b]})' for b in bin_order], fontsize=9)
        ax.set_ylabel('Percentage of SFP lines (%)')
        ax.set_xlabel('Partner age relative to main speaker')
        ax.set_title(
            f'{folder_id}: SFP Bucket % by Partner Relative Age — {partner_gender} opponents')
        ax.legend()

        plot_path = Path(output_dir) / f'sfp_ai_relative_age_{file_tag}_{folder_id}.svg'
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved relative-age chart: {plot_path.name}")


TOP_LEVEL_FOLDER_PATTERN = re.compile(r'^[A-Z]\d{3}$')  # e.g. C001, K001, S001, T001, W001


def find_cejc_top_level_folders(cejc_root: str = CEJC_ROOT) -> List[str]:
    """List every top-level conversation folder (e.g. C001, K001, S001, T001, W001) present
    in the local corpus. Excludes non-conversation folders like Himawari_CEJC."""
    return sorted(
        p.name for p in Path(cejc_root).iterdir()
        if p.is_dir() and TOP_LEVEL_FOLDER_PATTERN.match(p.name)
    )


def classify_and_plot_folder(counter: 'SFPCounter', folder_id: str, output_dir: Path) -> pd.DataFrame:
    """Classify one top-level folder's 1-on-1 conversations and write its CSV + charts
    into output_dir. Shared by the single-folder path and the full-corpus sweep."""
    print(f"\n{'=' * 60}")
    print(f"Classifying folder: {folder_id}")
    print('=' * 60)

    df_ai = counter.classify_folder_with_ai(folder_id)
    if df_ai.empty:
        print(f"No SFP candidates classified for {folder_id}.")
        return df_ai

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'sfp_ai_classification_{folder_id}.csv'
    df_ai.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"\nClassified {len(df_ai)} SFP candidates across "
          f"{df_ai['conversation_id'].nunique()} conversations in {folder_id}.")
    print("By bucket:")
    print(df_ai['ai_bucket'].value_counts().to_string())
    print(f"Saved AI classification CSV to: {out_path}")

    plot_ai_bucket_distribution(df_ai, folder_id, str(output_dir))
    plot_ai_relative_age_by_gender(df_ai, folder_id, str(output_dir))

    return df_ai


def build_overall_ai_summary(ai_output_dir: str = AI_OUTPUT_DIR) -> pd.DataFrame:
    """Pool every per-folder sfp_ai_classification_<folder>.csv under ai_output_dir into
    one combined DataFrame, tagged with a 'gender_age_pair' column (main gender/age vs
    partner gender/age) for the cross-conversation overall chart."""
    csv_paths = sorted(
        p for p in Path(ai_output_dir).glob('*/sfp_ai_classification_*.csv')
        if p.parent.name != 'overall'
    )
    if not csv_paths:
        print(f"No per-folder AI classification CSVs found under {ai_output_dir}")
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    if {'main_gender', 'main_age', 'partner_gender', 'partner_age'}.issubset(df.columns):
        df['gender_age_pair'] = (
            df['main_gender'] + ' ' + df['main_age'].astype(str)
            + ' × ' + df['partner_gender'] + ' ' + df['partner_age'].astype(str)
        )
    return df


def plot_ai_overall_distribution(df_ai: pd.DataFrame, output_dir: str):
    """Overall (all classified folders pooled) charts: SFP bucket % per gender/age
    conversation pair, split into two figures — one for male main speakers, one for female —
    sorted so the shift in composition across pairs is easy to see.

    Normalized per conversation: when multiple conversations share the same gender_age_pair,
    each conversation's bucket % is computed independently first, then averaged across
    conversations in the pair — so a long conversation doesn't outweigh a short one."""
    required_cols = {'gender_age_pair', 'main_gender', 'conversation_id', 'ai_bucket'}
    if df_ai.empty or not required_cols.issubset(df_ai.columns):
        print("No gender/age pair data available for the overall chart.")
        return

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    bucket_order = ['masculine', 'neutral', 'feminine']
    colors = {'masculine': '#3498db', 'neutral': '#95a5a6', 'feminine': '#e74c3c'}

    df = df_ai[df_ai['ai_bucket'].notna()]
    gender_groups = [('男性', 'male_speakers'), ('女性', 'female_speakers')]

    for main_gender, file_tag in gender_groups:
        sub = df[df['main_gender'] == main_gender]
        if sub.empty:
            continue

        # Bucket % within each conversation first, then average across conversations
        # sharing the same gender_age_pair (equal weight per conversation, not per SFP line).
        per_conv_pct = (
            sub.groupby(['gender_age_pair', 'conversation_id'])['ai_bucket']
            .value_counts(normalize=True)
            .unstack(fill_value=0.0) * 100
        ).reindex(columns=bucket_order, fill_value=0.0)
        pct_table = per_conv_pct.groupby('gender_age_pair').mean()
        # Sort pairs by feminine % so the composition shift reads left-to-right.
        pct_table = pct_table.sort_values('feminine')
        n_per_pair = sub.groupby('gender_age_pair')['conversation_id'].nunique().reindex(pct_table.index)

        fig, ax = plt.subplots(figsize=(max(10, len(pct_table) * 0.9), 6))
        bottom = pd.Series(0.0, index=pct_table.index)
        for bucket in bucket_order:
            values = pct_table[bucket]
            bars = ax.bar(pct_table.index, values, bottom=bottom, label=bucket, color=colors[bucket])
            for bar, pct in zip(bars, values):
                if pct > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                             f'{pct:.1f}%', ha='center', va='center', fontsize=8, color='white')
            bottom += values

        ax.set_ylabel('Percentage of SFP lines (%)')
        ax.set_xlabel('Main speaker × partner (gender, age)')
        ax.set_xticks(range(len(pct_table.index)))
        ax.set_xticklabels(
            [f'{p}\n({n_per_pair[p]} conv.)' for p in pct_table.index], rotation=45, ha='right', fontsize=8)
        ax.set_title(
            f'Overall SFP Gender Bucket % by Conversation Pair — {main_gender} main speakers')
        ax.legend()

        plot_path = Path(output_dir) / f'sfp_ai_overall_distribution_{file_tag}.svg'
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved overall distribution chart: {plot_path.name}")


RELATIVE_AGE_CATEGORIES = [
    'かなり年下',  # Much Younger: 15+ years younger
    '年下',       # Younger: 5-14 years younger
    '同年代',     # Same Age: +/- 4 years
    '年上',       # Older: 5-14 years older
    'かなり年上',  # Much Older: 15+ years older
]


def _relative_age_category(partner_age: str, main_age: str) -> Optional[str]:
    """Bucket (partner_age - main_age) into 5 coarse categories: much younger (15+),
    younger (5-14), same age (+/-4), older (5-14), much older (15+)."""
    try:
        partner_start = int(str(partner_age).split('-')[0])
        main_start = int(str(main_age).split('-')[0])
    except (ValueError, IndexError):
        return None
    diff = partner_start - main_start
    if diff <= -15:
        return 'かなり年下'
    if diff <= -5:
        return '年下'
    if diff <= 4:
        return '同年代'
    if diff <= 14:
        return '年上'
    return 'かなり年上'


def plot_ai_overall_relative_age(df_ai: pd.DataFrame, output_dir: str):
    """Pooled-across-conversations chart: two 100% stacked area charts (male opponents,
    female opponents) showing SFP bucket % across 5 coarse partner-relative-age categories
    (much younger / younger / same age / older / much older), using every classified
    conversation rather than a single folder.

    Normalized per conversation: each conversation's bucket % per category is computed
    independently first, then averaged across conversations sharing that category — so a
    long conversation doesn't outweigh a short one."""
    required_cols = {'partner_age', 'main_age', 'ai_bucket', 'partner_gender', 'main_gender', 'conversation_id'}
    if df_ai.empty or not required_cols.issubset(df_ai.columns):
        print("No relative-age data available for the overall relative-age chart.")
        return

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    df = df_ai.copy()
    df['relative_age_category'] = df.apply(
        lambda r: _relative_age_category(r['partner_age'], r['main_age']), axis=1)
    df = df[df['relative_age_category'].notna() & df['ai_bucket'].notna()]

    bucket_order = ['masculine', 'neutral', 'feminine']
    colors = {'masculine': '#3498db', 'neutral': '#95a5a6', 'feminine': '#e74c3c'}

    opponent_groups = [('男性', 'male_opponents'), ('女性', 'female_opponents')]
    main_gender_groups = [('男性', '男性 main speakers'), ('女性', '女性 main speakers')]

    for partner_gender, file_tag in opponent_groups:
        sub_opponent = df[df['partner_gender'] == partner_gender]
        if sub_opponent.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, (main_gender, panel_title) in zip(axes, main_gender_groups):
            sub = sub_opponent[sub_opponent['main_gender'] == main_gender]
            cat_order = [c for c in RELATIVE_AGE_CATEGORIES if c in sub['relative_age_category'].unique()]
            if not cat_order:
                ax.set_visible(False)
                continue

            per_conv_pct = (
                sub.groupby(['relative_age_category', 'conversation_id'])['ai_bucket']
                .value_counts(normalize=True)
                .unstack(fill_value=0.0) * 100
            ).reindex(columns=bucket_order, fill_value=0.0)
            pct_table = per_conv_pct.groupby('relative_age_category').mean().reindex(index=cat_order)
            n_conv_per_cat = sub.groupby('relative_age_category')['conversation_id'].nunique().reindex(cat_order)

            x = range(len(cat_order))
            ax.stackplot(
                x,
                [pct_table[bucket].values for bucket in bucket_order],
                labels=bucket_order,
                colors=[colors[b] for b in bucket_order],
            )

            ax.set_xticks(list(x))
            ax.set_xticklabels(
                [f'{c}\n({n_conv_per_cat[c]} conv.)' for c in cat_order], fontsize=9)
            ax.set_xlim(0, len(cat_order) - 1)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Partner age relative to main speaker')
            ax.set_title(panel_title)

        axes[0].set_ylabel('Percentage of SFP lines (%)')
        visible_axes = [ax for ax in axes if ax.get_visible()]
        if visible_axes:
            handles, labels = visible_axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3)
        fig.suptitle(f'Overall SFP Bucket % by Partner Relative Age — {partner_gender} opponents')

        plot_path = Path(output_dir) / f'sfp_ai_overall_relative_age_{file_tag}.svg'
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved overall relative-age chart: {plot_path.name}")


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
GENDER_INTENSITY_LEXICON_WITH_KA = (
    '/home/ryuu/Ryu/System_implement/LLM_Response/'
    'SFP_Bucket_by_gender_detailed.json'
)
DEFAULT_GENDER_INTENSITY_LEXICON = GENDER_INTENSITY_LEXICON_WITH_KA


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


def _classify_gender_intensity(entry: str, token_to_tier: Dict[str, str]) -> str:
    """Map one ai_matched_entry value to a gender-intensity tier using the flattened
    lexicon. Falls back to token-level voting (splitting the entry on '、') so a
    slightly different compound grouping in the data still resolves; unmatched
    entries (dialect forms, or particles dropped from the lexicon, e.g. か) land in
    'none'."""
    if not isinstance(entry, str) or not entry:
        return 'none'
    entry_n = _normalize_sfp_entry(entry)
    normalized_lookup = {_normalize_sfp_entry(k): v for k, v in token_to_tier.items()}
    if entry_n in normalized_lookup:
        return normalized_lookup[entry_n]

    votes: Dict[str, int] = defaultdict(int)
    for tok in entry.split('、'):
        tok_n = _normalize_sfp_entry(tok.strip())
        if tok_n in normalized_lookup:
            votes[normalized_lookup[tok_n]] += 1
    if votes:
        return max(votes.items(), key=lambda kv: kv[1])[0]
    return 'none'


def build_gender_intensity_summary(
    ai_output_dir: str = AI_OUTPUT_DIR,
    lexicon_path: str = DEFAULT_GENDER_INTENSITY_LEXICON,
    exclude_particles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Pool every sfp_ai_classification_*.csv found (recursively) under ai_output_dir,
    reclassify each row's ai_matched_entry into the 6-tier gender-intensity lexicon,
    and aggregate to one row per (speaker_age, speaker_gender) with tier percentages.
    Percentages are per-utterance (not per-conversation-averaged) across every pooled row.

    exclude_particles: raw marked_particle values (e.g. ['か']) to drop from the pool
    entirely before computing percentages — removed from N as well as the numerator,
    not folded into 'none'. Use this for a strict include/exclude comparison of one
    particle's effect on the rest of the distribution."""
    csv_paths = sorted(
        p for p in Path(ai_output_dir).glob('**/sfp_ai_classification_*.csv')
        if p.parent.name != 'overall'
    )
    if not csv_paths:
        print(f"No AI classification CSVs found under {ai_output_dir}")
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    required_cols = {'speaker_age', 'speaker_gender', 'ai_matched_entry', 'ai_bucket'}
    if not required_cols.issubset(df.columns):
        print(f"Pooled data is missing required columns: {required_cols - set(df.columns)}")
        return pd.DataFrame()

    if exclude_particles:
        df = df[~df['marked_particle'].isin(exclude_particles)]

    token_to_tier = _load_gender_intensity_lexicon(lexicon_path)

    def tier_for_row(row) -> str:
        if row['ai_bucket'] == 'none':
            return 'none'
        return _classify_gender_intensity(row.get('ai_matched_entry', ''), token_to_tier)

    df = df[df['speaker_age'].notna() & df['speaker_gender'].notna()]
    df['gender_intensity_tier'] = df.apply(tier_for_row, axis=1)

    pct_table = (
        df.groupby(['speaker_age', 'speaker_gender'])['gender_intensity_tier']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0) * 100
    ).reindex(columns=GENDER_INTENSITY_TIERS, fill_value=0.0)
    n_table = df.groupby(['speaker_age', 'speaker_gender']).size()

    result = pct_table.reset_index()
    result['n'] = result.apply(
        lambda r: n_table.loc[(r['speaker_age'], r['speaker_gender'])], axis=1)
    return result


def plot_gender_intensity_by_age(summary: pd.DataFrame, output_dir: str, variant_suffix: str = ''):
    """Two stacked-bar SVGs (one per speaker gender: female, male), each showing the
    6-tier gender-intensity composition of SFP usage across CEJC age brackets, pooled
    across every classified conversation. Mirrors the age-ordered layout used
    elsewhere in this module (youngest to oldest bracket, left to right).

    variant_suffix (e.g. '_with_ka' / '_no_ka') is appended to both the output
    filename and the chart title, so the included-vs-excluded-か versions don't
    overwrite each other and are distinguishable at a glance."""
    if summary.empty:
        print("No gender-intensity summary available to plot.")
        return

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'IPAGothic', 'Hiragino Maru Gothic Pro', 'Meiryo', 'sans-serif']

    age_order = ['20-24歳', '25-29歳', '30-34歳', '35-39歳', '40-44歳', '45-49歳',
                 '50-54歳', '55-59歳', '60-64歳', '65-69歳', '70-74歳']
    gender_groups = [('女性', 'female_speakers'), ('男性', 'male_speakers')]

    for speaker_gender, file_tag in gender_groups:
        sub = summary[summary['speaker_gender'] == speaker_gender]
        if sub.empty:
            continue
        # Always draw every age bracket in the fixed order (not just brackets with
        # data for this gender) — a missing bracket becomes an empty placeholder bar
        # below rather than a silently skipped column, so a reader isn't misled into
        # thinking that age simply wasn't part of the study.
        sub = sub.set_index('speaker_age').reindex(age_order)
        missing_ages = sub[sub['n'].isna()].index.tolist()

        fig, ax = plt.subplots(figsize=(max(8, len(sub) * 0.9), 6))
        bottom = pd.Series(0.0, index=sub.index)
        for tier in GENDER_INTENSITY_TIERS:
            values = sub[tier].fillna(0.0)
            bars = ax.bar(
                sub.index, values, bottom=bottom,
                label=GENDER_INTENSITY_LABELS[tier], color=GENDER_INTENSITY_COLORS[tier])
            for bar, pct in zip(bars, values):
                if pct >= 5:
                    text_color = '#0b0b0b' if tier in ('neutral', 'moderately_masculine', 'none') else 'white'
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                             f'{pct:.1f}%', ha='center', va='center', fontsize=8, color=text_color)
            bottom += values

        for age in missing_ages:
            idx = sub.index.get_loc(age)
            ax.bar(age, 100, color='#f2f1ec', hatch='///', edgecolor='#c3c2b7', linewidth=0.5, zorder=1)
            ax.text(idx, 50, 'No speaker\nin this age range', ha='center', va='center',
                     fontsize=7.5, color='#898781', rotation=90, zorder=2)

        n_per_age = sub['n']
        ax.set_ylabel('Percentage of SFP-marked utterances (%)')
        ax.set_xlabel('Speaker age bracket')
        ax.set_xticks(range(len(sub.index)))
        ax.set_xticklabels(
            [f'{a.replace("歳", "")}\n(N={int(n_per_age[a]):,})' if pd.notna(n_per_age[a]) else f'{a.replace("歳", "")}\n(N=0)'
             for a in sub.index],
            rotation=0, fontsize=8)
        ax.set_ylim(0, 100)
        title_tag_map = {'_with_ka': ' (か included)', '_no_ka': ' (か excluded)'}
        title_tag = title_tag_map.get(variant_suffix, f' ({variant_suffix.strip("_").replace("_", " ")})' if variant_suffix else '')
        ax.set_title(f'SFP Gender-Intensity by Age — {speaker_gender} speakers{title_tag}')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

        plot_path = Path(output_dir) / f'sfp_gender_intensity_by_age_{file_tag}{variant_suffix}.svg'
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Saved gender-intensity chart: {plot_path.name}")


def main():
    parser = argparse.ArgumentParser(description='CEJC Sentence Final Particle Counter')
    parser.add_argument(
        '--csv-output',
        action='store_true',
        help='Generate CSV output files in addition to Excel and HTML outputs.',
    )
    parser.add_argument(
        '--sfp-mode',
        choices=['old', 'new', 'ai'],
        default=SFP_MODE,
        help=(
            '"old" = original SFP set (default). '
            '"new" = gendered taxonomy from the 3-column table (algorithm-detectable subset only; '
            'items requiring LLM/prosody context are excluded). '
            '"ai" = Gemini classification against the full gendered taxonomy '
            '(pilot, single conversation via --conversation; requires GENAI_API_KEY).'
        ),
    )
    parser.add_argument(
        '--folder',
        default=DEFAULT_AI_FOLDER,
        help=(
            f'Top-level CEJC folder to classify in --sfp-mode ai (default: {DEFAULT_AI_FOLDER}). '
            'Only 1-on-1 (2-speaker) conversations under this folder are used.'
        ),
    )
    parser.add_argument(
        '--overall',
        action='store_true',
        help=(
            'With --sfp-mode ai: instead of classifying a folder, pool every already-classified '
            'folder under output/ai_pilot/ and build the overall cross-conversation chart. '
            'Run this once all intended top-level folders (C001, C002, ...) have been classified.'
        ),
    )
    parser.add_argument(
        '--sweep-all',
        action='store_true',
        help=(
            'With --sfp-mode ai: classify every top-level CEJC folder found locally '
            '(C001, C002, ...) in one run, then build the overall summary. '
            'Output goes to output/ai_pilot/<run timestamp>/{C001,C002,...,overall}/. '
            'Per-conversation results are cached (output/ai_cache/), so an interrupted '
            'sweep can be resumed with --run-dir instead of re-paying for finished work.'
        ),
    )
    parser.add_argument(
        '--run-dir',
        default=None,
        help=(
            'With --sfp-mode ai --sweep-all: resume into an existing timestamped run '
            'directory (e.g. output/ai_pilot/20260904_120000) instead of starting a new '
            'one. Folders/conversations already cached are skipped automatically.'
        ),
    )
    parser.add_argument(
        '--gender-intensity',
        action='store_true',
        help=(
            'With --sfp-mode ai: pool every already-classified sfp_ai_classification_*.csv '
            'found under --run-dir (or output/ai_pilot/ if omitted), reclassify each row '
            'into the 6-tier strongly/moderately feminine-masculine lexicon, and save two '
            'SVG bar charts (female/male speakers, by age bracket) into <run-dir>/overall/. '
            'Generates BOTH a か-included version and a か-excluded version (か rows '
            'dropped from the pool entirely, not folded into "none") in one run.'
        ),
    )
    parser.add_argument(
        '--lexicon-path',
        default=GENDER_INTENSITY_LEXICON_WITH_KA,
        help=(
            'Path to the {tier: [entry, ...]} JSON lexicon used by --gender-intensity '
            f'(default: {GENDER_INTENSITY_LEXICON_WITH_KA}). か stays in this lexicon '
            "(moderately_masculine) — the か-excluded variant is produced by skipping "
            "か rows outright (see build_gender_intensity_summary's exclude_particles), "
            "not by using a different lexicon."
        ),
    )
    args = parser.parse_args()

    if args.sfp_mode == 'ai' and args.gender_intensity:
        print("=" * 60)
        print("CEJC SFP Counter — mode: ai (gender-intensity summary)")
        print("=" * 60)
        source_dir = args.run_dir if args.run_dir else AI_OUTPUT_DIR
        overall_dir = Path(source_dir) / 'overall'
        overall_dir.mkdir(parents=True, exist_ok=True)

        # か included vs か excluded (dropped from N entirely, not folded into 'none').
        variants = [
            ('_with_ka', None),
            ('_no_ka', ['か']),
        ]

        for variant_suffix, exclude_particles in variants:
            tag = 'か included' if exclude_particles is None else 'か excluded'
            print(f"\n--- {tag} ---")
            summary = build_gender_intensity_summary(
                source_dir, args.lexicon_path, exclude_particles=exclude_particles)
            if summary.empty:
                continue
            summary_csv = overall_dir / f'sfp_gender_intensity_by_age{variant_suffix}.csv'
            summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            print(f"Saved pooled tier summary to: {summary_csv}")
            plot_gender_intensity_by_age(summary, str(overall_dir), variant_suffix)
        return

    if args.sfp_mode == 'ai' and args.sweep_all:
        run_dir = Path(args.run_dir) if args.run_dir else (
            Path(AI_OUTPUT_DIR) / datetime.now().strftime('%Y%m%d_%H%M%S'))
        folders = find_cejc_top_level_folders()
        print("=" * 60)
        print(f"CEJC SFP Counter — mode: ai (full corpus sweep)")
        print(f"Run output: {run_dir}{' (resuming)' if args.run_dir else ''}")
        print(f"Folders found: {', '.join(folders) if folders else '(none)'}")
        print("=" * 60)

        if not folders:
            print(f"No top-level C### folders found under {CEJC_ROOT}")
            return

        counter = SFPCounter()
        for folder_id in folders:
            classify_and_plot_folder(counter, folder_id, run_dir / folder_id)

        print(f"\n{'=' * 60}")
        print("Building overall summary across all swept folders")
        print("=" * 60)
        df_overall = build_overall_ai_summary(str(run_dir))
        if df_overall.empty:
            return
        overall_dir = run_dir / 'overall'
        overall_dir.mkdir(parents=True, exist_ok=True)
        overall_csv = overall_dir / 'sfp_ai_classification_overall.csv'
        df_overall.to_csv(overall_csv, index=False, encoding='utf-8-sig')
        print(f"Saved pooled CSV to: {overall_csv}")
        plot_ai_overall_distribution(df_overall, str(overall_dir))
        plot_ai_overall_relative_age(df_overall, str(overall_dir))

        print(f"\nSweep complete. All output under: {run_dir}")
        return

    if args.sfp_mode == 'ai' and args.overall:
        print("=" * 60)
        print("CEJC SFP Counter — mode: ai (overall summary)")
        print("=" * 60)
        df_overall = build_overall_ai_summary()
        if df_overall.empty:
            return
        overall_dir = Path(AI_OUTPUT_DIR) / 'overall'
        overall_dir.mkdir(parents=True, exist_ok=True)
        overall_dir_csv = overall_dir / 'sfp_ai_classification_overall.csv'
        df_overall.to_csv(overall_dir_csv, index=False, encoding='utf-8-sig')
        print(f"Saved pooled CSV to: {overall_dir_csv}")
        plot_ai_overall_distribution(df_overall, str(overall_dir))
        plot_ai_overall_relative_age(df_overall, str(overall_dir))
        return

    if args.sfp_mode == 'ai':
        print("=" * 60)
        print(f"CEJC SFP Counter — mode: ai (Gemini pilot, folder={args.folder})")
        print("=" * 60)

        counter = SFPCounter()
        folder_output_dir = Path(AI_OUTPUT_DIR) / args.folder
        df_ai = classify_and_plot_folder(counter, args.folder, folder_output_dir)
        if df_ai.empty:
            return

        print(
            "\nOnce every intended top-level folder has been classified, run "
            "`--sfp-mode ai --overall` to build the pooled cross-conversation summary, "
            "or use `--sfp-mode ai --sweep-all` to do the whole corpus + overall in one run."
        )
        return

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
