"""
CEJC F0 Adaptation Analysis — ΔF0 (Delta F0)

Research question: Does a speaker's F0 shift (relative to their own baseline)
correlate with their conversation partner's F0, age, or gender?

ΔF0 = speaker's F0 in this conversation − speaker's own baseline F0
Baseline = median F0 across all the speaker's conversations.

Uses per-speaker WAV files from the CEJC corpus, restricted to the focal
speaker's own turn segments (from transUnit CSVs) to avoid microphone
bleed-through from the partner's voice.

Analyzes only core speakers (primary corpus participants) in 1-on-1 conversations.
"""

import json
import logging
import re
import sys
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Use a CJK font so Japanese labels render correctly
_jp_font = fm.findfont(fm.FontProperties(family='Noto Sans CJK JP'))
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
from scipy import stats
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.outliers_influence import OLSInfluence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Paths / cache
CORPUS_ROOT   = Path('/home/ryuu/Ryu/Corpus/CEJC')
OUTPUT_DIR    = Path('/home/ryuu/Ryu/Dataset_Analysis/CEJC_F0_Analyze/output')
CACHE_FILE    = OUTPUT_DIR / 'f0_cache.json'
ENCODING      = 'shift_jis'
N_WORKERS     = 12

# Audio / F0 extraction thresholds
MIN_SEG_SECS  = 0.256   # min segment duration for pyin (4096 samples at 16 kHz)
MIN_VOICED    = 10      # discard speaker/conv with fewer voiced frames than this
FRAME_LENGTH  = 2048
HOP_LENGTH    = 256

# Gender-specific pyin F0 search range
FMIN_FEMALE   = 100.0
FMAX_FEMALE   = 500.0
FMIN_MALE     = 75.0
FMAX_MALE     = 350.0
FMIN_DEFAULT  = 75.0
FMAX_DEFAULT  = 500.0

# Age / relationship thresholds
AGE_REL_THRESHOLD = 5   # years; ±5 yr boundary for older/same/younger

# Plotting thresholds
N_WARN = 10             # cells with n < N_WARN get gray background + ⚠ in scatter plots
MIN_CONVS_PER_SPEAKER = 4  # speakers with fewer conversations are excluded from per-speaker plot


# ---------------------------------------------------------------------------
# Plot ordering / label constants
# ---------------------------------------------------------------------------

AGE_BUCKET_ORDER    = ['under20', '20s', '30s', '40s', '50s', '60s', '70+', 'unknown']
AGE_DIFF_BAND_ORDER = ['10+ younger', '5–9 younger', 'same (±5)', '5–9 older', '10+ older']
REL_AGE_ORDER       = ['younger', 'same', 'older']
GENDER_LABELS       = {'女性': 'Female', '男性': 'Male'}


def _gender_label(g):
    return GENDER_LABELS.get(g, str(g))


# ---------------------------------------------------------------------------
# Scale configuration — captures what differs between Hz-scale and
# semitone/z-scored-scale treatments of the same plots and statistics.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleConfig:
    name:         str    # 'hz' | 'semitone' — used in log messages only
    y_col:        str    # 'delta_f0' | 'delta_f0_semitone'
    x_col:        str    # 'partner_f0_median' | 'partner_f0_z'  (scatter-plot x)
    y_label:      str    # y-axis label
    x_label:      str    # x-axis label (scatter plots)
    title_suffix: str    # appended to figure suptitles
    out_dir_attr: str    # 'plots_dir' | 'semitone_plots_dir'
    reg_min_n:    int    # minimum n before drawing a regression line
    stats_header: str    # header line for the statistics report
    formula_lhs:  str    # left-hand side used in echoed regression-formula lines


HZ_CONFIG = ScaleConfig(
    name='hz',
    y_col='delta_f0',
    x_col='partner_f0_median',
    y_label='ΔF0 (Hz, relative to speaker baseline)',
    x_label='Partner F0 median (Hz)',
    title_suffix='',
    out_dir_attr='plots_dir',
    reg_min_n=3,
    stats_header='ΔF0 ~ partner F0',
    formula_lhs='ΔF0',
)

SEMITONE_CONFIG = ScaleConfig(
    name='semitone',
    y_col='delta_f0_semitone',
    x_col='partner_f0_z',
    y_label='ΔF0 (semitones)',
    x_label='Partner F0 (z-score within gender)',
    title_suffix='\n[semitone scale; partner F0 z-scored within gender]',
    out_dir_attr='semitone_plots_dir',
    reg_min_n=N_WARN,
    stats_header='ΔF0 (semitones) ~ partner F0 (z-score within gender)',
    formula_lhs='delta_f0_semitone',
)


# ---------------------------------------------------------------------------
# F0 Cache — skip WAV extraction when results already exist
# ---------------------------------------------------------------------------

def load_cache(cache_file):
    """Load cached F0 results from a previous run. Returns empty dict if none."""
    if not cache_file.exists():
        return {}
    with open(cache_file, encoding='utf-8') as f:
        raw = json.load(f)
    # Keys stored as "conv_id|ic_label" — restore to tuple keys
    return {tuple(k.split('|', 1)): v for k, v in raw.items()}


def save_cache(f0_results, cache_file):
    """Persist F0 results dict to JSON for reuse in future runs."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {'|'.join(k): v for k, v in f0_results.items()}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
    logger.info(f'[cache] Saved {len(f0_results)} entries → {cache_file}')


# ---------------------------------------------------------------------------
# Step 1: Metadata loading
# ---------------------------------------------------------------------------

def _parse_age_mid(age_raw):
    """Parse "40-44歳" → 42.0. Returns None on failure."""
    if pd.isna(age_raw):
        return None
    m = re.match(r'(\d+)-(\d+)歳', str(age_raw))
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2.0
    return None


def _age_diff_band(d):
    """Bin age_diff (partner − speaker) into relative-age band label."""
    if d is None or pd.isna(d):
        return None
    if d <= -10:
        return '10+ younger'
    elif d <= -5:
        return '5–9 younger'
    elif d < 5:
        return 'same (±5)'
    elif d < 10:
        return '5–9 older'
    else:
        return '10+ older'


def _parse_age_decade(age_mid):
    """Bin age midpoint into decade label for plots."""
    if age_mid is None:
        return 'unknown'
    a = int(age_mid)
    if a < 20:
        return 'under20'
    elif a < 30:
        return '20s'
    elif a < 40:
        return '30s'
    elif a < 50:
        return '40s'
    elif a < 60:
        return '50s'
    elif a < 70:
        return '60s'
    else:
        return '70+'


def load_metadata(corpus_root):
    """
    Load the three CEJC metadata CSVs plus relationship data.

    Returns:
        speaker_meta: dict[speaker_id -> {age_raw, age_mid, gender, relationship}]
        conv_speaker_pairs: dict[conv_id -> {core: {speaker_id, ic_label},
                                              partner: {speaker_id, ic_label}}]
        conv_relationship: dict[conv_id -> {relationship_raw, relationship_primary}]
    """
    # Conversation.csv: filter to 2-speaker conversations + collect relationship
    conv_df = pd.read_csv(corpus_root / 'Conversation.csv', encoding=ENCODING)
    two_spk_ids = set(conv_df.loc[conv_df['話者数'] == 2, '会話ID'])
    logger.info(f'[metadata] 2-speaker conversations: {len(two_spk_ids)}')

    conv_relationship = {}
    for _, row in conv_df.iterrows():
        raw = row.get('話者間の関係性', None)
        if pd.isna(raw):
            raw = None
        primary = raw.split('・')[0].strip() if raw else None
        conv_relationship[row['会話ID']] = {
            'relationship_raw':     raw,
            'relationship_primary': primary,
        }

    # Speaker_data.csv: build speaker metadata dict (includes fine-grained relationship)
    spk_df = pd.read_csv(corpus_root / 'Speaker_data.csv', encoding=ENCODING)
    speaker_meta = {}
    for _, row in spk_df.iterrows():
        sid = row['話者ID']
        age_raw = row.get('年齢', None)
        rel_raw = row.get('協力者からみた関係性', None)
        speaker_meta[sid] = {
            'age_raw':      age_raw,
            'age_mid':      _parse_age_mid(age_raw),
            'gender':       row.get('性別', None),
            'relationship': None if pd.isna(rel_raw) else rel_raw,
        }
    logger.info(f'[metadata] Speakers in Speaker_data.csv: {len(speaker_meta)}')

    # Speaker_Conversation_Relation.csv: build conv -> core/partner pairs
    rel_df = pd.read_csv(corpus_root / 'Speaker_Conversation_Relation.csv', encoding=ENCODING)
    # Keep only IC-prefixed labels (exclude Z-prefix bystanders)
    rel_df = rel_df[rel_df['話者ラベル'].str.match(r'^IC')]
    # Keep only rows for 2-speaker conversations
    rel_df = rel_df[rel_df['会話ID'].isin(two_spk_ids)]

    conv_speaker_pairs = {}
    skipped = 0
    for conv_id, grp in rel_df.groupby('会話ID'):
        if len(grp) != 2:
            logger.warning(f'[skip] {conv_id}: expected 2 IC speakers, found {len(grp)}')
            skipped += 1
            continue

        # Core speaker: 話者ID with no underscore (e.g. "T001")
        # Partner speaker: 話者ID with underscore (e.g. "T001_001" or "T002_011")
        core_rows    = grp[~grp['話者ID'].str.contains('_')]
        partner_rows = grp[grp['話者ID'].str.contains('_')]

        if len(core_rows) != 1 or len(partner_rows) != 1:
            logger.warning(f'[skip] {conv_id}: cannot identify unique core/partner')
            skipped += 1
            continue

        core_row    = core_rows.iloc[0]
        partner_row = partner_rows.iloc[0]

        conv_speaker_pairs[conv_id] = {
            'core':    {'speaker_id': core_row['話者ID'],    'ic_label': core_row['話者ラベル']},
            'partner': {'speaker_id': partner_row['話者ID'], 'ic_label': partner_row['話者ラベル']},
        }

    logger.info(f'[metadata] Valid conv pairs: {len(conv_speaker_pairs)}  |  skipped: {skipped}')
    return speaker_meta, conv_speaker_pairs, conv_relationship


# ---------------------------------------------------------------------------
# Step 2: Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(conv_id, corpus_root, ic_label):
    """
    Resolve WAV and transUnit CSV paths for a given conv_id and IC label.

    For segmented conv_ids (e.g. T002_011a), the directory uses the base
    conv_id without the suffix (T002_011/) but filenames include the full
    conv_id with suffix.

    Returns (wav_path, trans_path) or (None, None) if files are missing.
    """
    base_id  = re.sub(r'[abc]$', '', conv_id)
    conv_dir = corpus_root / conv_id[:4] / base_id

    m = re.match(r'^(IC\d+)_', ic_label)
    if not m:
        logger.warning(f'[warn] Cannot parse IC prefix from label: {ic_label}')
        return None, None
    ic_prefix = m.group(1)

    wav_path   = conv_dir / f'{conv_id}_{ic_prefix}.wav'
    trans_path = conv_dir / f'{conv_id}-transUnit.csv'

    if not wav_path.exists():
        logger.warning(f'[missing] WAV: {wav_path}')
        return None, None
    if not trans_path.exists():
        logger.warning(f'[missing] transUnit: {trans_path}')
        return None, None

    return wav_path, trans_path


# ---------------------------------------------------------------------------
# Step 3: F0 extraction (multiprocessing-compatible — takes a single tuple)
# ---------------------------------------------------------------------------

def extract_f0_for_speaker(args):
    """
    Extract F0 statistics for one speaker in one conversation.

    Uses only the speaker's own turn segments from transUnit CSV to avoid
    contamination from microphone bleed-through of the partner's voice.

    Args:
        args: (conv_id, wav_path_str, trans_path_str, ic_label, fmin, fmax)

    Returns:
        dict with F0 statistics, or {'valid': False, 'reason': str}
    """
    conv_id, wav_path_str, trans_path_str, ic_label, fmin, fmax = args
    wav_path   = Path(wav_path_str)
    trans_path = Path(trans_path_str)

    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)
    except Exception as e:
        return {'conv_id': conv_id, 'ic_label': ic_label, 'valid': False,
                'reason': f'WAV load error: {e}'}

    try:
        trans_df = pd.read_csv(trans_path, encoding=ENCODING)
    except Exception as e:
        return {'conv_id': conv_id, 'ic_label': ic_label, 'valid': False,
                'reason': f'transUnit load error: {e}'}

    m = re.match(r'^(IC\d+)_', ic_label)
    if not m:
        return {'conv_id': conv_id, 'ic_label': ic_label, 'valid': False,
                'reason': 'Cannot parse IC prefix'}
    ic_prefix = m.group(1)

    speaker_segs = trans_df[trans_df['speakerID'].str.startswith(ic_prefix)]

    all_f0      = []
    total_voiced = 0
    total_frames = 0
    n_used       = 0

    for _, row in speaker_segs.iterrows():
        dur = row['endTime'] - row['startTime']
        if dur < MIN_SEG_SECS:
            continue
        start_sample = int(row['startTime'] * sr)
        end_sample   = int(row['endTime']   * sr)
        segment = y[start_sample:end_sample]
        if len(segment) < FRAME_LENGTH:
            continue

        try:
            f0, voiced_flag, _ = librosa.pyin(
                segment, fmin=fmin, fmax=fmax, sr=sr,
                frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
            )
        except Exception:
            continue

        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
        all_f0.extend(valid_f0.tolist())
        total_voiced += int(voiced_flag.sum())
        total_frames += len(voiced_flag)
        n_used += 1

    if total_voiced < MIN_VOICED:
        return {'conv_id': conv_id, 'ic_label': ic_label, 'valid': False,
                'reason': f'Too few voiced frames: {total_voiced}'}

    all_f0 = np.array(all_f0)
    return {
        'conv_id':          conv_id,
        'ic_label':         ic_label,
        'f0_median':        float(np.median(all_f0)),
        'f0_mean':          float(np.mean(all_f0)),
        'f0_sd':            float(np.std(all_f0)),
        'f0_q25':           float(np.percentile(all_f0, 25)),
        'f0_q75':           float(np.percentile(all_f0, 75)),
        'voiced_frames':    total_voiced,
        'total_frames':     total_frames,
        'voiced_ratio':     total_voiced / total_frames if total_frames > 0 else 0.0,
        'n_segments_used':  n_used,
        'valid':            True,
    }


# ---------------------------------------------------------------------------
# Step 4: Build results DataFrame
# ---------------------------------------------------------------------------

def build_results_df(conv_speaker_pairs, speaker_meta, f0_results, conv_relationship):
    """
    Combine F0 results with speaker demographics into a single DataFrame.
    One row per valid (core speaker, conversation) pair.

    Adds:
      partner_f0_median    — partner's F0 in this conversation
      f0_baseline          — core speaker's median F0 across all their conversations
      baseline_n_convs     — number of conversations used for the baseline
      delta_f0             — f0_median - f0_baseline
      relationship_raw     — full 話者間の関係性 string (e.g. 友人知人・家族)
      relationship_primary — first token before ・ (e.g. 友人知人)
      partner_relationship — fine-grained 協力者からみた関係性 for the partner speaker
    """
    rows = []
    for conv_id, pair in conv_speaker_pairs.items():
        core    = pair['core']
        partner = pair['partner']

        key = (conv_id, core['ic_label'])
        if key not in f0_results:
            continue
        result = f0_results[key]

        core_meta    = speaker_meta.get(core['speaker_id'], {})
        partner_meta = speaker_meta.get(partner['speaker_id'], {})

        core_age_mid    = core_meta.get('age_mid')
        partner_age_mid = partner_meta.get('age_mid')

        if core_age_mid is not None and partner_age_mid is not None:
            age_diff = partner_age_mid - core_age_mid
            if age_diff >= AGE_REL_THRESHOLD:
                relative_age = 'older'
            elif age_diff <= -AGE_REL_THRESHOLD:
                relative_age = 'younger'
            else:
                relative_age = 'same'
        else:
            age_diff     = None
            relative_age = None

        # Partner F0 (NaN if extraction failed)
        partner_key = (conv_id, partner['ic_label'])
        partner_f0  = f0_results[partner_key]['f0_median'] if partner_key in f0_results else None

        # Relationship
        rel = conv_relationship.get(conv_id, {})

        rows.append({
            'conv_id':            conv_id,
            'conv_type':          conv_id[0],
            'core_speaker_id':    core['speaker_id'],
            'core_ic_label':      core['ic_label'],
            'gender':             core_meta.get('gender'),
            'age_raw':            core_meta.get('age_raw'),
            'age_mid':            core_age_mid,
            'partner_id':         partner['speaker_id'],
            'partner_gender':     partner_meta.get('gender'),
            'partner_age_raw':    partner_meta.get('age_raw'),
            'partner_age_mid':    partner_age_mid,
            'age_diff':           age_diff,
            'age_diff_band':      _age_diff_band(age_diff),
            'relative_age':       relative_age,
            'partner_age_bucket': _parse_age_decade(partner_age_mid),
            'partner_f0_median':    partner_f0,
            'relationship_raw':     rel.get('relationship_raw'),
            'relationship_primary': rel.get('relationship_primary'),
            'partner_relationship': partner_meta.get('relationship'),
            'f0_median':            result['f0_median'],
            'f0_mean':            result['f0_mean'],
            'f0_sd':              result['f0_sd'],
            'f0_q25':             result['f0_q25'],
            'f0_q75':             result['f0_q75'],
            'voiced_frames':      result['voiced_frames'],
            'voiced_ratio':       result['voiced_ratio'],
            'n_segments_used':    result['n_segments_used'],
        })

    df = pd.DataFrame(rows)

    # Compute per-speaker baseline (median F0 across all their conversations)
    baseline     = df.groupby('core_speaker_id')['f0_median'].median()
    baseline_n   = df.groupby('core_speaker_id')['f0_median'].count()
    df['f0_baseline']      = df['core_speaker_id'].map(baseline)
    df['baseline_n_convs'] = df['core_speaker_id'].map(baseline_n)
    df['delta_f0']         = df['f0_median'] - df['f0_baseline']
    df['delta_f0_semitone'] = np.where(
        df['f0_baseline'] > 0,
        12 * np.log2(df['f0_median'] / df['f0_baseline']),
        np.nan
    )

    logger.info(f'[results] Valid rows: {len(df)}')
    n_single = (df['baseline_n_convs'] == 1).sum()
    if n_single:
        logger.info(f'[results] {n_single} rows from speakers with only 1 conversation '
                     f'(delta_f0 will be 0 for these; excluded from delta analyses)')
    return df


# ---------------------------------------------------------------------------
# Step 5a: Statistics
# ---------------------------------------------------------------------------

def run_statistics(df_delta, output_dir, cfg: ScaleConfig, append: bool):
    """
    Three statistical checks on the ΔF0 ~ partner_F0 relationship, on the
    scale described by `cfg`:

    1. OLS per group: slope, p-value, R² for each (speaker_gender × partner_gender)
       combination — reveals whether the slope survives within each gender group.
    2. Interaction OLS: y ~ x * speaker_gender — formal test of whether
       the two slopes differ from each other.
    3. Mixed-effects model: y ~ x * speaker_gender + (1 | speaker_id)
       — accounts for non-independence from repeated speakers.
    """
    x_col, y_col = cfg.x_col, cfg.y_col
    df = df_delta.dropna(subset=[x_col, y_col]).copy()
    df['is_female'] = (df['gender_en'] == 'Female').astype(int)

    lines = []
    lines.append(('\n' if append else '') + '=' * 70)
    lines.append(f'STATISTICAL ANALYSIS: {cfg.stats_header}')
    lines.append('=' * 70)

    # ------------------------------------------------------------------
    # 1. OLS within each (speaker_gender × partner_gender) cell
    # ------------------------------------------------------------------
    lines.append('\n--- 1. OLS within speaker_gender × partner_gender cells ---')
    if not append:
        lines.append('  (checks whether slope persists inside each gender group)')
    lines.append(f'  {"Group":<30} {"n":>4}  {"slope":>8}  {"R²":>6}  {"p":>8}')
    lines.append('  ' + '-' * 62)
    for sg in ['Female', 'Male']:
        for pg in ['Female', 'Male']:
            sub = df[(df['gender_en'] == sg) & (df['partner_gender_en'] == pg)]
            if len(sub) < 5:
                continue
            slope, intercept, r, p, se = stats.linregress(sub[x_col], sub[y_col])
            label = f'{sg} speaker → {pg} partner'
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            lines.append(f'  {label:<30} {len(sub):>4}  {slope:>+8.4f}  {r**2:>6.3f}  {p:>8.4f} {sig}')

    # ------------------------------------------------------------------
    # 2. Interaction OLS: y ~ x * speaker_gender (plain OLS)
    # ------------------------------------------------------------------
    lines.append('\n--- 2. Interaction OLS (plain, ignores repeated speakers) ---')
    lines.append(f'  {cfg.formula_lhs} ~ {x_col} * C(gender_en)')
    try:
        m_ols = ols(f'{y_col} ~ {x_col} * C(gender_en)', data=df).fit()
        lines.append(m_ols.summary2().tables[1].to_string())
        lines.append(f'\n  Model R² = {m_ols.rsquared:.3f}  |  Adj. R² = {m_ols.rsquared_adj:.3f}')
        if not append:
            lines.append(f'  F({int(m_ols.df_model)}, {int(m_ols.df_resid)}) = {m_ols.fvalue:.2f}  p = {m_ols.f_pvalue:.4f}')
    except Exception as e:
        lines.append(f'  OLS failed: {e}')

    # ------------------------------------------------------------------
    # 3. Mixed-effects: y ~ x * speaker_gender + (1 | speaker_id)
    # ------------------------------------------------------------------
    lines.append('\n--- 3. Mixed-effects model (random intercept per speaker) ---')
    lines.append(f'  {cfg.formula_lhs} ~ {x_col} * C(gender_en) + (1 | core_speaker_id)')
    try:
        m_lme = mixedlm(
            f'{y_col} ~ {x_col} * C(gender_en)',
            data=df,
            groups=df['core_speaker_id']
        ).fit(reml=True)
        lines.append(m_lme.summary().tables[1].to_string())
        lines.append(f'\n  Log-likelihood: {m_lme.llf:.2f}')
    except Exception as e:
        lines.append(f'  Mixed-effects failed: {e}')

    lines.append('\n' + '=' * 70)

    report = '\n'.join(lines)
    logger.info(report)

    stats_path = output_dir / 'delta_f0_statistics.txt'
    mode = 'a' if append else 'w'
    with open(stats_path, mode, encoding='utf-8') as f:
        f.write(report)
    verb = 'appended' if append else 'saved'
    logger.info(f'[output] Statistics {verb}: {stats_path}')


# ---------------------------------------------------------------------------
# Step 5b: Outlier-analysis helpers (Cook's distance / cluster-robust SE)
# ---------------------------------------------------------------------------

def _cooks_influence_mask(sub, x_col, y_col, threshold_divisor=4):
    """
    Boolean mask of high-influence points (Cook's D > threshold_divisor/n)
    for an OLS fit of y_col ~ x_col on `sub`. All-False if n < 4 or the fit
    fails.
    """
    mask = np.zeros(len(sub), dtype=bool)
    if len(sub) < 4:
        return mask
    try:
        m = ols(f'{y_col} ~ {x_col}', data=sub).fit()
        cd = OLSInfluence(m).cooks_distance[0]
        mask = cd > (threshold_divisor / len(sub))
    except Exception:
        pass
    return mask


def _cluster_robust_fit(sub, x_col, y_col, group_col='core_speaker_id'):
    """
    Fit OLS y_col ~ x_col on `sub` and return (se, pvalue) for the slope
    using cluster-robust covariance (clustered on group_col). Returns
    (None, None) on failure.
    """
    try:
        m = ols(f'{y_col} ~ {x_col}', data=sub).fit()
        m_cl = m.get_robustcov_results(cov_type='cluster', groups=sub[group_col])
        return float(m_cl.bse[1]), float(m_cl.pvalues[1])
    except Exception:
        return None, None


def _sig(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


# ---------------------------------------------------------------------------
# Step 5c: Plot builder functions
# ---------------------------------------------------------------------------

def plot_boxplot_gender_pair(df_delta, cfg: ScaleConfig, out_dir):
    """ΔF0 by speaker gender x partner gender (2-panel boxplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, focal_gender in zip(axes, ['Female', 'Male']):
        sub = df_delta[df_delta['gender_en'] == focal_gender]
        if sub.empty:
            ax.set_title(f'{focal_gender} (no data)')
            continue
        sns.boxplot(data=sub, x='partner_gender_en', y=cfg.y_col,
                    hue='partner_gender_en', order=['Female', 'Male'],
                    ax=ax, palette='Set2', legend=False)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(f'Focal: {focal_gender}')
        ax.set_xlabel('Partner gender')
        ax.set_ylabel(cfg.y_label)
        for x_pos, label in enumerate(['Female', 'Male']):
            n = len(sub[sub['partner_gender_en'] == label])
            ax.text(x_pos, ax.get_ylim()[0], f'n={n}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'ΔF0 by speaker gender x partner gender{cfg.title_suffix}')
    fig.tight_layout()
    fig.savefig(out_dir / 'delta_f0_by_gender_pair.png', dpi=150)
    plt.close(fig)


def plot_boxplot_relative_age(df_delta, cfg: ScaleConfig, out_dir):
    """ΔF0 by relative partner age (±5 yr threshold)."""
    df_age = df_delta.dropna(subset=['relative_age'])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df_age, x='relative_age', y=cfg.y_col,
                hue='gender_en', order=REL_AGE_ORDER,
                hue_order=['Female', 'Male'], ax=ax, palette='Set1')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'ΔF0 by relative partner age (+-5 yr threshold){cfg.title_suffix}')
    ax.set_xlabel('Partner is... (relative to focal speaker)')
    ax.set_ylabel(cfg.y_label)
    ax.legend(title='Speaker gender')
    fig.tight_layout()
    fig.savefig(out_dir / 'delta_f0_by_relative_age.png', dpi=150)
    plt.close(fig)


def plot_pointplot_age_bucket(df_delta, cfg: ScaleConfig, out_dir):
    """ΔF0 by partner age bucket (mean +- SD lineplot)."""
    valid_buckets = [b for b in AGE_BUCKET_ORDER if b in df_delta['partner_age_bucket'].values]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(data=df_delta, x='partner_age_bucket', y=cfg.y_col,
                  hue='gender_en', order=valid_buckets,
                  hue_order=['Female', 'Male'],
                  errorbar='sd', ax=ax, palette='Set1',
                  markers=['o', 's'], linestyles=['-', '--'])
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'ΔF0 by partner age group (mean +- SD){cfg.title_suffix}')
    ax.set_xlabel('Partner age group')
    ax.set_ylabel(cfg.y_label)
    ax.legend(title='Speaker gender')
    fig.tight_layout()
    fig.savefig(out_dir / 'delta_f0_by_partner_age_bucket.png', dpi=150)
    plt.close(fig)


def plot_boxplot_relationship(df_delta, cfg: ScaleConfig, out_dir):
    """ΔF0 by speaker-partner relationship type (boxplot, columns ordered by count)."""
    df_rel = df_delta.dropna(subset=['relationship_primary', cfg.y_col])
    if df_rel.empty:
        return
    rel_order = (df_rel.groupby('relationship_primary').size()
                 .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(max(8, len(rel_order) * 1.6), 5))
    sns.boxplot(data=df_rel, x='relationship_primary', y=cfg.y_col,
                hue='gender_en', order=rel_order,
                hue_order=['Female', 'Male'], ax=ax, palette='Set1')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'ΔF0 by speaker–partner relationship type{cfg.title_suffix}')
    ax.set_xlabel('Relationship (primary category)')
    ax.set_ylabel(cfg.y_label)
    ax.legend(title='Speaker gender')
    for i, rel in enumerate(rel_order):
        for j, sg_en in enumerate(['Female', 'Male']):
            n = len(df_rel[(df_rel['relationship_primary'] == rel) &
                           (df_rel['gender_en'] == sg_en)])
            if n > 0:
                offset = -0.2 + j * 0.4
                ax.text(i + offset, ax.get_ylim()[1] * 0.97,
                        f'n={n}', ha='center', va='top', fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_dir / 'delta_f0_by_relationship.png', dpi=150)
    plt.close(fig)


_PARTNER_COLORS = {'Female': '#e41a1c', 'Male': '#377eb8'}
_SPEAKER_COLORS = {'Female': '#e41a1c', 'Male': '#377eb8'}


def plot_scatter_by_gender(df_delta, cfg: ScaleConfig, out_dir, outlier_analysis: bool):
    """
    ΔF0 ~ partner F0 — faceted by speaker gender, coloured by partner gender.
    Each panel shows within-partner-gender slopes to distinguish categorical
    from continuous effects.

    When outlier_analysis is True, additionally computes Cook's-distance
    high-influence points (hollow diamond markers), cluster-robust SEs, and
    a sensitivity refit with high-influence points dropped.
    """
    x_col, y_col = cfg.x_col, cfg.y_col
    df_scatter = df_delta.dropna(subset=[x_col, y_col])
    if df_scatter.empty:
        return

    figsize = (14, 7) if outlier_analysis else (13, 6)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, sg in zip(axes, ['Female', 'Male']):
        sub_sg = df_scatter[df_scatter['gender_en'] == sg].copy()
        slope_blocks = []

        for pg, color in _PARTNER_COLORS.items():
            sub_pg = sub_sg[sub_sg['partner_gender_en'] == pg].copy().reset_index(drop=True)
            if sub_pg.empty:
                continue

            if outlier_analysis:
                infl_mask = _cooks_influence_mask(sub_pg, x_col, y_col)

                normal = sub_pg[~infl_mask]
                if not normal.empty:
                    ax.scatter(normal[x_col], normal[y_col],
                               color=color, alpha=0.6, s=35, label=pg)
                outliers = sub_pg[infl_mask]
                if not outliers.empty:
                    ax.scatter(outliers[x_col], outliers[y_col],
                               color=color, alpha=0.9, s=80, marker='D',
                               facecolors='none', edgecolors=color, linewidths=1.5)

                if len(sub_pg) < N_WARN:
                    continue

                sns.regplot(data=sub_pg, x=x_col, y=y_col,
                            scatter=False, ax=ax, color=color,
                            line_kws={'linewidth': 2, 'linestyle': '--'})

                sl, _, _, p_ols, se_ols = stats.linregress(sub_pg[x_col], sub_pg[y_col])

                se_cl, p_cl = _cluster_robust_fit(sub_pg, x_col, y_col)
                cl_str = f'  cluster: SE={se_cl:.3f} {_sig(p_cl)}' if se_cl is not None else '  cluster: n/a'

                if infl_mask.sum() > 0:
                    sub_drop = sub_pg[~infl_mask]
                    sl2, _, _, p2, _ = stats.linregress(sub_drop[x_col], sub_drop[y_col])
                    sens_str = f'  −◇: β={sl2:+.3f} {_sig(p2)}'
                    se_cl_drop, p_cl_drop = _cluster_robust_fit(sub_drop, x_col, y_col)
                    combined_str = (f'  −◇+cluster: SE={se_cl_drop:.3f} {_sig(p_cl_drop)}'
                                     if se_cl_drop is not None else '  −◇+cluster: n/a')
                else:
                    sens_str = ''
                    combined_str = ''

                n_infl = int(infl_mask.sum())
                infl_note = f' [◇×{n_infl}]' if n_infl else ''
                block = (f'{pg[0]}: β={sl:+.3f} SE={se_ols:.3f} {_sig(p_ols)}{infl_note}\n'
                         f'{cl_str}')
                if sens_str:
                    block += f'\n{sens_str}'
                if combined_str:
                    block += f'\n{combined_str}'
                slope_blocks.append(block)
            else:
                sns.scatterplot(data=sub_pg, x=x_col, y=y_col,
                                color=color, alpha=0.6, ax=ax, label=pg)
                if len(sub_pg) > 2:
                    sns.regplot(data=sub_pg, x=x_col, y=y_col,
                                scatter=False, ax=ax, color=color,
                                line_kws={'linewidth': 2, 'linestyle': '--'})
                    slope = np.polyfit(sub_pg[x_col], sub_pg[y_col], 1)[0]
                    slope_blocks.append(f'{pg[0]}: 傾き={slope:+.3f}')

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_title(f'Focal speaker: {sg}')
        ax.set_xlabel(cfg.x_label)
        ax.set_ylabel(cfg.y_label if sg == 'Female' else '')
        if outlier_analysis:
            handles, labels = ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = h
            ax.legend(seen.values(), seen.keys(), title='Partner gender')
        else:
            ax.legend(title='Partner gender')
        pg_counts = '\n'.join(f'{pg[0]}: n={len(sub_sg[sub_sg["partner_gender_en"]==pg])}'
                              for pg in ['Female', 'Male'])
        ax.annotate(pg_counts, xy=(0.97, 0.97),
                    xycoords='axes fraction', ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        if slope_blocks:
            sep = '\n\n' if outlier_analysis else '\n'
            fontsize = 7 if outlier_analysis else 8
            pad = 0.3 if outlier_analysis else 0.3
            ax.annotate(sep.join(slope_blocks), xy=(0.03, 0.03),
                        xycoords='axes fraction', ha='left', va='bottom', fontsize=fontsize,
                        bbox=dict(boxstyle=f'round,pad={pad}', fc='white', alpha=0.7))

    if outlier_analysis:
        fig.suptitle('ΔF0 vs. partner F0 — within-gender-group slopes'
                     f'{cfg.title_suffix}\n'
                     '◇ = high-influence (Cook\'s D > 4/n);  −◇ = slope after removing them',
                     fontsize=10)
    else:
        fig.suptitle('ΔF0 vs. partner F0 — within-gender-group slopes\n'
                     '(dashed lines = within-group OLS; dotted = ΔF0=0 baseline)')
    fig.tight_layout()
    fig.savefig(out_dir / 'delta_f0_vs_partner_f0.png', dpi=150)
    plt.close(fig)


def plot_scatter_by_gender_cleaned(df_delta, cfg: ScaleConfig, out_dir):
    """
    Same as plot_scatter_by_gender(outlier_analysis=True) but with
    high-influence points (Cook's D > 4/n) REMOVED before plotting/fitting,
    with cluster-robust SE annotated. Semitone-scale only.
    """
    x_col, y_col = cfg.x_col, cfg.y_col
    df_scatter = df_delta.dropna(subset=[x_col, y_col])
    if df_scatter.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, sg in zip(axes, ['Female', 'Male']):
        sub_sg = df_scatter[df_scatter['gender_en'] == sg].copy()
        slope_blocks = []

        for pg, color in _PARTNER_COLORS.items():
            sub_pg = sub_sg[sub_sg['partner_gender_en'] == pg].copy().reset_index(drop=True)
            if sub_pg.empty:
                continue

            infl_mask = _cooks_influence_mask(sub_pg, x_col, y_col)
            sub_clean = sub_pg[~infl_mask].copy()
            n_dropped = int(infl_mask.sum())

            if not sub_clean.empty:
                ax.scatter(sub_clean[x_col], sub_clean[y_col],
                           color=color, alpha=0.6, s=35, label=pg)

            if len(sub_clean) < N_WARN:
                continue

            sns.regplot(data=sub_clean, x=x_col, y=y_col,
                        scatter=False, ax=ax, color=color,
                        line_kws={'linewidth': 2, 'linestyle': '-'})

            sl_c, _, _, p_ols_c, se_ols_c = stats.linregress(sub_clean[x_col], sub_clean[y_col])

            se_cl_c, p_cl_c = _cluster_robust_fit(sub_clean, x_col, y_col)
            cl_str_c = f'  cluster: SE={se_cl_c:.3f} {_sig(p_cl_c)}' if se_cl_c is not None else '  cluster: n/a'

            drop_note = f' [−◇×{n_dropped}]' if n_dropped else ''
            block_c = (f'{pg[0]}: β={sl_c:+.3f} SE={se_ols_c:.3f} {_sig(p_ols_c)}{drop_note}\n'
                       f'{cl_str_c}')
            slope_blocks.append(block_c)

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_title(f'Focal speaker: {sg}')
        ax.set_xlabel(cfg.x_label)
        ax.set_ylabel(cfg.y_label if sg == 'Female' else '')
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), title='Partner gender')
        pg_counts = '\n'.join(f'{pg[0]}: n={len(sub_sg[sub_sg["partner_gender_en"]==pg])}'
                              for pg in ['Female', 'Male'])
        ax.annotate(pg_counts, xy=(0.97, 0.97),
                    xycoords='axes fraction', ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        if slope_blocks:
            ax.annotate('\n\n'.join(slope_blocks), xy=(0.03, 0.03),
                        xycoords='axes fraction', ha='left', va='bottom', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    fig.suptitle('ΔF0 vs. partner F0 — high-influence points removed (Cook\'s D > 4/n)'
                 f'{cfg.title_suffix}\n'
                 'Solid line = OLS on cleaned data; SE from cluster-robust (by speaker)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / 'delta_f0_vs_partner_f0_cleaned.png', dpi=150)
    plt.close(fig)


def plot_scatter_faceted(df_delta, cfg: ScaleConfig, out_dir, *,
                          facet_col, facet_order, out_name, title, warn_gate: bool):
    """
    ΔF0 ~ partner F0 — 2-row x N-col grid faceted by `facet_col`: rows =
    speaker gender, columns = facet value, colour = partner gender. Used for
    both "by relative age band" (facet_col='age_diff_band', facet_order=
    AGE_DIFF_BAND_ORDER) and "by relationship type" (facet_col=
    'relationship_primary', facet_order=None — falls back to descending
    frequency order).

    Row split + partner-gender colouring keeps dot colour consistent with
    plot_scatter_by_gender / plot_scatter_grid_by_age (colour always means
    partner gender), rather than switching meaning between figures.
    """
    x_col, y_col = cfg.x_col, cfg.y_col
    df_facet = df_delta.dropna(subset=[x_col, y_col, facet_col])
    if df_facet.empty:
        return

    if facet_order is not None:
        cols = [c for c in facet_order if c in df_facet[facet_col].values]
    else:
        cols = (df_facet.groupby(facet_col).size()
                .sort_values(ascending=False).index.tolist())
    if not cols:
        return

    n_cols = len(cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 10),
                              sharey=True, sharex=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    legend_handles = []
    for col_i, facet_val in enumerate(cols):
        sub_facet = df_facet[df_facet[facet_col] == facet_val]
        for row_i, sg in enumerate(['Female', 'Male']):
            ax = axes[row_i, col_i]
            sub = sub_facet[sub_facet['gender_en'] == sg]
            n_total = len(sub)
            slope_annotations = []
            for pg, color in _PARTNER_COLORS.items():
                sub_pg = sub[sub['partner_gender_en'] == pg]
                if sub_pg.empty:
                    continue
                sc = ax.scatter(sub_pg[x_col], sub_pg[y_col],
                                color=color, alpha=0.6, s=30, label=pg)
                if col_i == 0 and row_i == 0:
                    legend_handles.append(sc)
                if len(sub_pg) >= cfg.reg_min_n:
                    sns.regplot(data=sub_pg, x=x_col, y=y_col,
                                scatter=False, ax=ax, color=color,
                                line_kws={'linewidth': 1.8, 'linestyle': '--'})
                    slope = np.polyfit(sub_pg[x_col], sub_pg[y_col], 1)[0]
                    slope_annotations.append(f'{pg[0]}: {slope:+.3f}')
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
            if warn_gate and n_total < N_WARN:
                ax.set_facecolor('#f0f0f0')
            if row_i == 0:
                ax.set_title(str(facet_val), fontsize=10)
            ax.set_xlabel(cfg.x_label if row_i == 1 else '', fontsize=9)
            ax.set_ylabel(f'{sg}\n{cfg.y_label}' if col_i == 0 else '', fontsize=9)
            warn = ' ⚠' if (warn_gate and n_total < N_WARN) else ''
            pg_counts = '\n'.join(f'{pg[0]}: n={len(sub[sub["partner_gender_en"]==pg])}'
                                  for pg in ['Female', 'Male']) + warn
            ax.annotate(pg_counts, xy=(0.97, 0.97),
                        xycoords='axes fraction', ha='right', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            if slope_annotations:
                slope_txt = '傾き\n' + '\n'.join(slope_annotations)
                ax.annotate(slope_txt, xy=(0.03, 0.03),
                            xycoords='axes fraction', ha='left', va='bottom', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    fig.suptitle(f'{title}{cfg.title_suffix}', fontsize=11)
    if legend_handles:
        fig.legend(legend_handles, list(_PARTNER_COLORS.keys()),
                   title='Partner gender', loc='lower center', ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.12, top=0.88, hspace=0.3, wspace=0.15)
    fig.savefig(out_dir / out_name, dpi=150)
    plt.close(fig)


def plot_scatter_grid_by_age(df_delta, cfg: ScaleConfig, out_dir, *, warn_gate: bool):
    """
    ΔF0 ~ partner F0 — 2-row x N-col grid: rows = speaker gender,
    cols = partner age bucket, colour = partner gender.
    """
    x_col, y_col = cfg.x_col, cfg.y_col
    df_scatter = df_delta.dropna(subset=[x_col, y_col])
    if df_scatter.empty:
        return

    valid_buckets = [b for b in AGE_BUCKET_ORDER
                     if b in df_scatter['partner_age_bucket'].values and b != 'unknown']
    n_cols = len(valid_buckets)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8),
                             sharey=True, sharex=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    legend_handles = []
    for col_i, bucket in enumerate(valid_buckets):
        for row_i, sg in enumerate(['Female', 'Male']):
            ax = axes[row_i, col_i]
            sub = df_scatter[
                (df_scatter['gender_en'] == sg) &
                (df_scatter['partner_age_bucket'] == bucket)
            ]
            n_total = len(sub)
            slope_lines = []
            for pg, color in _PARTNER_COLORS.items():
                sub_pg = sub[sub['partner_gender_en'] == pg]
                if sub_pg.empty:
                    continue
                sc = ax.scatter(sub_pg[x_col], sub_pg[y_col],
                                color=color, alpha=0.6, s=30, label=pg)
                if col_i == 0 and row_i == 0:
                    legend_handles.append(sc)
                if len(sub_pg) >= cfg.reg_min_n:
                    sns.regplot(data=sub_pg, x=x_col, y=y_col,
                                scatter=False, ax=ax, color=color,
                                line_kws={'linewidth': 1.5, 'linestyle': '--'})
                    slope = np.polyfit(sub_pg[x_col], sub_pg[y_col], 1)[0]
                    slope_lines.append(f'{pg[0]}: {slope:+.3f}')
            ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
            if warn_gate and n_total < N_WARN:
                ax.set_facecolor('#f0f0f0')
            warn = ' ⚠' if (warn_gate and n_total < N_WARN) else ''
            pg_counts = '\n'.join(f'{pg[0]}: n={len(sub[sub["partner_gender_en"]==pg])}'
                                  for pg in ['Female', 'Male']) + warn
            ax.annotate(pg_counts, xy=(0.97, 0.97),
                        xycoords='axes fraction', ha='right', va='top', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            if slope_lines:
                slope_txt = '傾き\n' + '\n'.join(slope_lines)
                ax.annotate(slope_txt, xy=(0.03, 0.03),
                            xycoords='axes fraction', ha='left', va='bottom', fontsize=6,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            if row_i == 0:
                ax.set_title(bucket, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f'{sg}\n{cfg.y_label}', fontsize=8)
            else:
                ax.set_ylabel('')
            if row_i == 1:
                ax.set_xlabel(cfg.x_label, fontsize=8)
            else:
                ax.set_xlabel('')
            ax.tick_params(labelsize=7)

    fig.suptitle('ΔF0 vs. partner F0 — faceted by partner age group'
                 f'{cfg.title_suffix}', fontsize=11)
    if legend_handles:
        fig.legend(legend_handles, list(_PARTNER_COLORS.keys()),
                   title='Partner gender', loc='lower center', ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.15)
    fig.savefig(out_dir / 'delta_f0_vs_partner_f0_by_age.png', dpi=150)
    plt.close(fig)


def _speaker_grid(df_delta, x_col, y_col, x_label, y_label, title_suffix, out_path):
    """Small-multiples grid: one panel per speaker, colored by partner gender."""
    counts = df_delta.groupby('core_speaker_id').size()
    valid_ids = counts[counts >= MIN_CONVS_PER_SPEAKER].index.tolist()
    spk_info = (df_delta[['core_speaker_id', 'gender_en']]
                .drop_duplicates()
                .set_index('core_speaker_id'))
    valid_ids.sort(key=lambda s: (spk_info.loc[s, 'gender_en'], s))

    # Shared y-limits: p2/p98 of the plotted data + 10% padding (clips extreme outliers)
    plotted_y = df_delta[df_delta['core_speaker_id'].isin(valid_ids)][y_col].dropna()
    y_lo = float(np.percentile(plotted_y, 2))
    y_hi = float(np.percentile(plotted_y, 98))
    pad = (y_hi - y_lo) * 0.10
    y_lim = (y_lo - pad, y_hi + pad)

    n_spk = len(valid_ids)
    n_cols = 6
    n_rows = (n_spk + n_cols - 1) // n_cols
    partner_colors = {'Female': '#e41a1c', 'Male': '#377eb8'}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows),
                             sharey=True, sharex=True)
    axes_flat = np.array(axes).flatten()

    legend_handles = []
    for idx, spk_id in enumerate(valid_ids):
        ax = axes_flat[idx]
        sub = df_delta[df_delta['core_speaker_id'] == spk_id].dropna(subset=[x_col, y_col])
        gender_initial = sub['gender_en'].iloc[0][0]
        slope_lines = []
        for pg, color in partner_colors.items():
            sub_pg = sub[sub['partner_gender_en'] == pg]
            if sub_pg.empty:
                continue
            sc = ax.scatter(sub_pg[x_col], sub_pg[y_col],
                            color=color, alpha=0.7, s=35, label=pg)
            if idx == 0:
                legend_handles.append(sc)
            if len(sub_pg) >= 3:
                sns.regplot(data=sub_pg, x=x_col, y=y_col,
                            scatter=False, ax=ax, color=color,
                            line_kws={'linewidth': 1.5, 'linestyle': '--'})
                slope = np.polyfit(sub_pg[x_col], sub_pg[y_col], 1)[0]
                slope_lines.append(f'{pg[0]}: {slope:+.2f}')
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_ylim(y_lim)
        ax.set_title(f'{spk_id} ({gender_initial})', fontsize=8)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel(y_label if idx % n_cols == 0 else '', fontsize=7)
        ax.tick_params(labelsize=6)
        if slope_lines:
            ax.annotate('傾き\n' + '\n'.join(slope_lines), xy=(0.03, 0.03),
                        xycoords='axes fraction', ha='left', va='bottom', fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
        pg_counts = '\n'.join(
            f'{pg[0]}: n={len(sub[sub["partner_gender_en"] == pg])}'
            for pg in ['Female', 'Male']
        )
        ax.annotate(pg_counts, xy=(0.97, 0.97),
                    xycoords='axes fraction', ha='right', va='top', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    for ax in axes_flat[n_spk:]:
        ax.set_visible(False)

    fig.suptitle(f'ΔF0 per speaker — {title_suffix}\n'
                 f'(color = partner gender; dashed = per-gender OLS if n≥3; Female→Male)',
                 fontsize=11)
    if legend_handles:
        fig.legend(legend_handles, list(partner_colors.keys()),
                   title='Partner gender', loc='lower center', ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.08, hspace=0.55, wspace=0.15)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _forest_plot_slopes(df_delta, x_col, y_col, x_label, title, out_path):
    """Forest / caterpillar plot: per-speaker OLS slopes ± 95% CI, split by partner gender."""
    from scipy import stats as scipy_stats

    counts = df_delta.groupby('core_speaker_id').size()
    valid_ids = counts[counts >= MIN_CONVS_PER_SPEAKER].index.tolist()
    spk_info = (df_delta[['core_speaker_id', 'gender_en']]
                .drop_duplicates()
                .set_index('core_speaker_id'))

    def _overall_slope(spk_id):
        s = df_delta[df_delta['core_speaker_id'] == spk_id].dropna(subset=[x_col, y_col])
        if len(s) < 3:
            return 0.0
        return float(np.polyfit(s[x_col], s[y_col], 1)[0])

    valid_ids.sort(key=_overall_slope)  # ascending: lowest at bottom, highest at top

    partner_colors = {'Female': '#e41a1c', 'Male': '#377eb8'}
    offsets = {'Female': +0.15, 'Male': -0.15}

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(valid_ids) + 2))

    legend_handles = []
    legend_labels = []
    for row_i, spk_id in enumerate(valid_ids):
        sub = df_delta[df_delta['core_speaker_id'] == spk_id].dropna(subset=[x_col, y_col])
        for pg, color in partner_colors.items():
            sub_pg = sub[sub['partner_gender_en'] == pg]
            if len(sub_pg) < 3:
                continue
            slope, _, _, _, se = scipy_stats.linregress(sub_pg[x_col], sub_pg[y_col])
            ci = se * scipy_stats.t.ppf(0.975, df=len(sub_pg) - 2)
            y_pos = row_i + offsets[pg]
            eb = ax.errorbar(slope, y_pos, xerr=ci,
                             fmt='o', color=color, capsize=3,
                             markersize=5, linewidth=1.2, alpha=0.85)
            if pg not in legend_labels:
                legend_handles.append(eb)
                legend_labels.append(pg)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(valid_ids)))
    ax.set_yticklabels(
        [f'{s} ({spk_info.loc[s, "gender_en"][0]})' for s in valid_ids],
        fontsize=8
    )
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_title(title, fontsize=10)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, title='Partner gender', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 5d: run_analysis() orchestration
# ---------------------------------------------------------------------------

def run_analysis(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    hz_plots_dir = plots_dir / 'hz'
    hz_plots_dir.mkdir(exist_ok=True)
    semitone_plots_dir = plots_dir / 'semitone_zscored'
    semitone_plots_dir.mkdir(exist_ok=True)

    df = df.copy()
    df['gender_en']         = df['gender'].map(_gender_label)
    df['partner_gender_en'] = df['partner_gender'].map(_gender_label)

    # z-score partner F0 within partner gender — removes bimodal gender effect from x-axis
    df['partner_f0_z'] = df.groupby('partner_gender_en')['partner_f0_median'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Save full results CSV
    csv_path = output_dir / 'f0_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f'[output] Saved: {csv_path}  ({len(df)} rows)')

    # Subset for delta analyses: exclude speakers with only 1 conversation
    df_delta = df[df['baseline_n_convs'] >= 2].copy()

    # Statistical tests (Hz scale then semitone scale)
    run_statistics(df_delta, output_dir, HZ_CONFIG, append=False)
    run_statistics(df_delta, output_dir, SEMITONE_CONFIG, append=True)

    # Console summary
    logger.info('\n=== ΔF0 by speaker gender x partner gender ===')
    summary1 = df_delta.groupby(['gender_en', 'partner_gender_en'])['delta_f0'].agg(
        ['median', 'std', 'count']
    ).round(2)
    logger.info(summary1.to_string())

    logger.info('\n=== ΔF0 by speaker gender x relative partner age ===')
    df_age = df_delta.dropna(subset=['relative_age'])
    summary2 = df_age.groupby(['gender_en', 'relative_age'])['delta_f0'].agg(
        ['median', 'std', 'count']
    ).round(2)
    logger.info(summary2.to_string())

    logger.info('\n=== Pearson correlations with ΔF0 ===')
    df_corr = df_delta.dropna(subset=['partner_f0_median', 'delta_f0'])
    if len(df_corr) > 1:
        r_f0  = df_corr['delta_f0'].corr(df_corr['partner_f0_median'])
        logger.info(f'  ΔF0 ~ partner F0:  r = {r_f0:.3f}  (n={len(df_corr)})')
    df_corr2 = df_delta.dropna(subset=['partner_age_mid', 'delta_f0'])
    if len(df_corr2) > 1:
        r_age = df_corr2['delta_f0'].corr(df_corr2['partner_age_mid'])
        logger.info(f'  ΔF0 ~ partner age: r = {r_age:.3f}  (n={len(df_corr2)})')

    # Plots 1-3: Hz scale
    plot_boxplot_gender_pair(df_delta, HZ_CONFIG, hz_plots_dir)
    plot_boxplot_relative_age(df_delta, HZ_CONFIG, hz_plots_dir)
    plot_pointplot_age_bucket(df_delta, HZ_CONFIG, hz_plots_dir)

    # Plot 4: ΔF0 ~ partner F0 faceted by speaker gender
    plot_scatter_by_gender(df_delta, HZ_CONFIG, hz_plots_dir, outlier_analysis=False)

    # Plot 4b: faceted by relative age band
    plot_scatter_faceted(
        df_delta, HZ_CONFIG, hz_plots_dir,
        facet_col='age_diff_band', facet_order=AGE_DIFF_BAND_ORDER,
        out_name='delta_f0_vs_partner_f0_by_rel_age.png',
        title='ΔF0 vs. partner F0 — faceted by relative partner age\n'
              '(rows = speaker gender; color = partner gender; dashed = per-partner-gender OLS)',
        warn_gate=False,
    )

    # Plot 4c: faceted by relationship type
    plot_scatter_faceted(
        df_delta, HZ_CONFIG, hz_plots_dir,
        facet_col='relationship_primary', facet_order=None,
        out_name='delta_f0_vs_partner_f0_by_relationship.png',
        title='ΔF0 vs. partner F0 — faceted by relationship type\n'
              '(rows = speaker gender; color = partner gender; dashed = per-partner-gender OLS)',
        warn_gate=False,
    )

    # Plot 5: grid — speaker gender x partner age bucket
    plot_scatter_grid_by_age(df_delta, HZ_CONFIG, hz_plots_dir, warn_gate=False)

    # Console summary: ΔF0 by relationship
    df_rel = df_delta.dropna(subset=['relationship_primary'])
    if not df_rel.empty:
        logger.info('\n=== ΔF0 by speaker gender × relationship ===')
        summary_rel = df_rel.groupby(['gender_en', 'relationship_primary'])['delta_f0'].agg(
            ['median', 'std', 'count']
        ).round(2)
        logger.info(summary_rel.to_string())

    # Plot 6: ΔF0 by relationship type
    plot_boxplot_relationship(df_delta, HZ_CONFIG, hz_plots_dir)

    # -----------------------------------------------------------------------
    # Semitone / z-scored partner F0 versions — saved in semitone_zscored/
    # -----------------------------------------------------------------------
    plot_boxplot_gender_pair(df_delta, SEMITONE_CONFIG, semitone_plots_dir)
    plot_boxplot_relative_age(df_delta, SEMITONE_CONFIG, semitone_plots_dir)
    plot_pointplot_age_bucket(df_delta, SEMITONE_CONFIG, semitone_plots_dir)

    # S-Plot 4: with outlier analysis (Cook's distance, cluster-robust SE, sensitivity refit)
    plot_scatter_by_gender(df_delta, SEMITONE_CONFIG, semitone_plots_dir, outlier_analysis=True)

    # S-Plot 4c "cleaned": high-influence points removed
    plot_scatter_by_gender_cleaned(df_delta, SEMITONE_CONFIG, semitone_plots_dir)

    # S-Plot 4b: faceted by relative age band
    plot_scatter_faceted(
        df_delta, SEMITONE_CONFIG, semitone_plots_dir,
        facet_col='age_diff_band', facet_order=AGE_DIFF_BAND_ORDER,
        out_name='delta_f0_vs_partner_f0_by_rel_age.png',
        title='ΔF0 vs. partner F0 — faceted by relative partner age',
        warn_gate=True,
    )

    # S-Plot 4c: faceted by relationship type
    plot_scatter_faceted(
        df_delta, SEMITONE_CONFIG, semitone_plots_dir,
        facet_col='relationship_primary', facet_order=None,
        out_name='delta_f0_vs_partner_f0_by_relationship.png',
        title='ΔF0 vs. partner F0 — faceted by relationship type',
        warn_gate=True,
    )

    # S-Plot 5: grid — speaker gender x partner age bucket
    plot_scatter_grid_by_age(df_delta, SEMITONE_CONFIG, semitone_plots_dir, warn_gate=True)

    # S-Plot 6: ΔF0 by relationship type (boxplot)
    plot_boxplot_relationship(df_delta, SEMITONE_CONFIG, semitone_plots_dir)

    # Plot 7: per-speaker small multiples
    _speaker_grid(df_delta,
                  x_col='partner_f0_median', y_col='delta_f0',
                  x_label='Partner F0 (Hz)', y_label='ΔF0 (Hz)',
                  title_suffix='Hz scale',
                  out_path=hz_plots_dir / 'delta_f0_by_speaker.png')

    _speaker_grid(df_delta,
                  x_col='partner_f0_z', y_col='delta_f0_semitone',
                  x_label='Partner F0 (z-score within gender)', y_label='ΔF0 (semitones)',
                  title_suffix='semitone scale; partner F0 z-scored within gender',
                  out_path=semitone_plots_dir / 'delta_f0_by_speaker.png')

    _forest_plot_slopes(
        df_delta,
        x_col='partner_f0_z', y_col='delta_f0_semitone',
        x_label='OLS slope: ΔF0 (semitones) per σ of partner F0',
        title='Per-speaker accommodation slopes ± 95% CI\n'
              '(ΔF0 semitones ~ partner F0 z-score, split by partner gender)',
        out_path=semitone_plots_dir / 'forest_slope_by_speaker.png'
    )

    logger.info(f'\n[output] Hz-scale plots saved to: {hz_plots_dir}')
    logger.info(f'[output] Semitone plots saved to: {semitone_plots_dir}')


# ---------------------------------------------------------------------------
# Step 6: Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    logger.info('=== CEJC F0 Adaptation Analysis ===\n')

    # Phase 1: Load metadata (now returns conv_relationship too)
    speaker_meta, conv_speaker_pairs, conv_relationship = load_metadata(CORPUS_ROOT)

    # Phase 2: Load cache + build task list
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = load_cache(CACHE_FILE)
    logger.info(f'[cache] Loaded {len(cached)} cached F0 entries from {CACHE_FILE}')

    def _fmin_fmax(gender):
        if gender == '女性':
            return FMIN_FEMALE, FMAX_FEMALE
        elif gender == '男性':
            return FMIN_MALE, FMAX_MALE
        return FMIN_DEFAULT, FMAX_DEFAULT

    tasks = []
    n_cached = 0
    for conv_id, pair in conv_speaker_pairs.items():
        for role in ('core', 'partner'):
            spk       = pair[role]
            key       = (conv_id, spk['ic_label'])
            if key in cached:
                n_cached += 1
                continue
            gender    = speaker_meta.get(spk['speaker_id'], {}).get('gender')
            fmin, fmax = _fmin_fmax(gender)
            wav_path, trans_path = resolve_paths(conv_id, CORPUS_ROOT, spk['ic_label'])
            if wav_path is None:
                continue
            tasks.append((conv_id, str(wav_path), str(trans_path),
                          spk['ic_label'], fmin, fmax))

    logger.info(f'[main] From cache: {n_cached}  |  Newly queued: {len(tasks)}')

    # Phase 3: Parallel F0 extraction (only for uncached tasks)
    new_results = {}
    if tasks:
        logger.info(f'[main] Running F0 extraction with {N_WORKERS} workers...\n')
        if N_WORKERS > 1:
            with multiprocessing.Pool(N_WORKERS) as pool:
                raw_results = pool.map(extract_f0_for_speaker, tasks)
        else:
            raw_results = [extract_f0_for_speaker(t) for t in tasks]

        n_invalid = 0
        for r in raw_results:
            if r.get('valid'):
                new_results[(r['conv_id'], r['ic_label'])] = r
            else:
                n_invalid += 1
                logger.warning(f'[invalid] {r["conv_id"]} {r["ic_label"]}: {r.get("reason", "")}')
        logger.info(f'[main] New extractions: {len(new_results)} valid  |  {n_invalid} invalid')
    else:
        logger.info('[main] All tasks served from cache — skipping WAV extraction.')

    # Merge and persist cache
    f0_results = {**cached, **new_results}
    save_cache(f0_results, CACHE_FILE)
    logger.info(f'[main] Total F0 results: {len(f0_results)}')

    # Phase 4: Build results DataFrame
    results_df = build_results_df(conv_speaker_pairs, speaker_meta, f0_results,
                                  conv_relationship)

    if results_df.empty:
        logger.info('[main] No valid results. Exiting.')
        return

    # Phase 5: Analysis and plots
    run_analysis(results_df, OUTPUT_DIR)

    logger.info('\n=== Done ===')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
