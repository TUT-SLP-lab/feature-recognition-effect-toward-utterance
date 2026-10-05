"""
CEJC F0 Dynamic Range Analysis

Research question (per professor feedback): imitating a partner's *average*
F0 is the wrong target — two speakers can share the same mean while one
stays flat and the other swings widely. The swing (dynamic range) is what
should be imitated: does a core speaker's F0 dynamic range track their
conversation partner's dynamic range in the same conversation?

Dynamic range is measured via IQR (f0_q75 - f0_q25), in semitones (log
scale) — robust to the octave errors / pyin search-bound artifacts that
make min-max range unusable (min-max was dropped after f0_min/f0_max were
found pinned at the pyin fmin/fmax search bounds for ~90%+ of rows).

Uses per-speaker WAV files from the CEJC corpus, restricted to each
speaker's own turn segments (from transUnit CSVs) to avoid microphone
bleed-through from the partner's voice. Same corpus scope as CEJC_F0.py:
core speakers in 1-on-1 (2-speaker) conversations. Both core and partner
speakers' dynamic range are extracted so that core-range can be regressed
against partner-range (mimicry test), analogous to CEJC_F0.py's
ΔF0-vs-partner-F0 analysis but for range instead of level.
"""

import json
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

fm.findfont(fm.FontProperties(family='Noto Sans CJK JP'))
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
from scipy import stats

from CEJC_F0 import (
    CORPUS_ROOT, ENCODING, N_WORKERS,
    MIN_SEG_SECS, MIN_VOICED, FRAME_LENGTH, HOP_LENGTH,
    FMIN_FEMALE, FMAX_FEMALE, FMIN_MALE, FMAX_MALE, FMIN_DEFAULT, FMAX_DEFAULT,
    GENDER_LABELS, _gender_label,
    load_metadata, resolve_paths,
)
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path('/home/ryuu/Ryu/Dataset_Analysis/CEJC_F0_Analyze/output')
CACHE_FILE  = OUTPUT_DIR / 'dynamic_range_cache.json'
RESULTS_CSV = OUTPUT_DIR / 'dynamic_range_results.csv'
STATS_FILE  = OUTPUT_DIR / 'dynamic_range_statistics.txt'
PLOTS_DIR   = OUTPUT_DIR / 'dynamic_range_plots'


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache(cache_file):
    if not cache_file.exists():
        return {}
    with open(cache_file, encoding='utf-8') as f:
        raw = json.load(f)
    return {tuple(k.split('|', 1)): v for k, v in raw.items()}


def save_cache(results, cache_file):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {'|'.join(k): v for k, v in results.items()}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
    logger.info(f'[cache] Saved {len(results)} entries -> {cache_file}')


# ---------------------------------------------------------------------------
# F0 range extraction (multiprocessing-compatible — takes a single tuple)
# ---------------------------------------------------------------------------

def extract_f0_range_for_speaker(args):
    """
    Extract F0 dynamic-range statistics for one speaker in one conversation.

    Uses only the speaker's own turn segments from transUnit CSV to avoid
    contamination from microphone bleed-through of the partner's voice.

    Args:
        args: (conv_id, wav_path_str, trans_path_str, ic_label, fmin, fmax)

    Returns:
        dict with F0 range statistics, or {'valid': False, 'reason': str}
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

    all_f0       = []
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
    f0_median = float(np.median(all_f0))
    f0_q25    = float(np.percentile(all_f0, 25))
    f0_q75    = float(np.percentile(all_f0, 75))

    f0_range_iqr_semitone = 12 * np.log2(f0_q75 / f0_q25) if f0_q25 > 0 else None
    f0_sd_semitone = float(np.std(12 * np.log2(all_f0 / f0_median))) if f0_median > 0 else None

    return {
        'conv_id':          conv_id,
        'ic_label':         ic_label,
        'f0_median':        f0_median,
        'f0_q25':           f0_q25,
        'f0_q75':           f0_q75,
        'f0_range_iqr':     f0_q75 - f0_q25,
        'f0_range_iqr_semitone': f0_range_iqr_semitone,
        'f0_sd_semitone':        f0_sd_semitone,
        'voiced_frames':    total_voiced,
        'total_frames':     total_frames,
        'voiced_ratio':     total_voiced / total_frames if total_frames > 0 else 0.0,
        'n_segments_used':  n_used,
        'valid':            True,
    }


# ---------------------------------------------------------------------------
# Build results DataFrame
# ---------------------------------------------------------------------------

def build_range_df(conv_speaker_pairs, speaker_meta, range_results):
    """
    One row per valid (core speaker, conversation) pair: core speaker's F0
    dynamic range, their and their partner's gender, and the partner's own
    dynamic range in that same conversation (for the range-mimicry test —
    does core range track partner range?).
    """
    rows = []
    for conv_id, pair in conv_speaker_pairs.items():
        core    = pair['core']
        partner = pair['partner']

        key = (conv_id, core['ic_label'])
        if key not in range_results:
            continue
        result = range_results[key]

        partner_key    = (conv_id, partner['ic_label'])
        partner_result = range_results.get(partner_key)

        core_meta    = speaker_meta.get(core['speaker_id'], {})
        partner_meta = speaker_meta.get(partner['speaker_id'], {})

        rows.append({
            'conv_id':          conv_id,
            'core_speaker_id':  core['speaker_id'],
            'core_ic_label':    core['ic_label'],
            'gender':           core_meta.get('gender'),
            'gender_en':        _gender_label(core_meta.get('gender')),
            'partner_id':       partner['speaker_id'],
            'partner_gender':   partner_meta.get('gender'),
            'partner_gender_en': _gender_label(partner_meta.get('gender')),
            'f0_q25':           result['f0_q25'],
            'f0_q75':           result['f0_q75'],
            'f0_range_iqr':     result['f0_range_iqr'],
            'f0_range_iqr_semitone':    result['f0_range_iqr_semitone'],
            'f0_sd_semitone':           result['f0_sd_semitone'],
            'partner_f0_range_iqr_semitone': partner_result['f0_range_iqr_semitone'] if partner_result else None,
            'partner_f0_sd_semitone':        partner_result['f0_sd_semitone'] if partner_result else None,
            'voiced_frames':    result['voiced_frames'],
            'voiced_ratio':     result['voiced_ratio'],
            'n_segments_used':  result['n_segments_used'],
        })

    df = pd.DataFrame(rows)
    logger.info(f'[results] Valid rows: {len(df)}')
    n_partner = df['partner_f0_range_iqr_semitone'].notna().sum()
    logger.info(f'[results] Rows with partner range available: {n_partner}')
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _sig(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def run_statistics(df, output_dir):
    lines = []
    lines.append('=' * 70)
    lines.append('DYNAMIC RANGE ANALYSIS: F0 range ~ speaker gender x partner gender')
    lines.append('=' * 70)

    for metric, label in [('f0_range_iqr_semitone', 'IQR range (semitones)'),
                           ('f0_sd_semitone', 'SD (semitones)')]:
        lines.append(f'\n--- {label} ---')
        lines.append(f'  {"Group":<28} {"n":>4}  {"mean":>7}  {"median":>7}  {"sd":>7}')
        lines.append('  ' + '-' * 58)
        for sg in ['Female', 'Male']:
            for pg in ['Female', 'Male']:
                sub = df[(df['gender_en'] == sg) & (df['partner_gender_en'] == pg)][metric].dropna()
                if len(sub) == 0:
                    continue
                label_str = f'{sg} speaker -> {pg} partner'
                lines.append(f'  {label_str:<28} {len(sub):>4}  {sub.mean():>7.3f}  '
                              f'{sub.median():>7.3f}  {sub.std():>7.3f}')

        lines.append(f'\n  --- t-test: Female-partner vs Male-partner range, within each speaker gender ---')
        for sg in ['Female', 'Male']:
            sub_f = df[(df['gender_en'] == sg) & (df['partner_gender_en'] == 'Female')][metric].dropna()
            sub_m = df[(df['gender_en'] == sg) & (df['partner_gender_en'] == 'Male')][metric].dropna()
            if len(sub_f) < 2 or len(sub_m) < 2:
                lines.append(f'  {sg} speaker: insufficient data (n_female_partner={len(sub_f)}, n_male_partner={len(sub_m)})')
                continue
            t, p = stats.ttest_ind(sub_f, sub_m, equal_var=False)
            lines.append(f'  {sg} speaker: t={t:+.3f}  p={p:.4f} {_sig(p)}  '
                          f'(Female partner n={len(sub_f)}, Male partner n={len(sub_m)})')

    # ------------------------------------------------------------------
    # Range mimicry: does core speaker's range track partner's range?
    # ------------------------------------------------------------------
    for metric, partner_metric, label in [
        ('f0_range_iqr_semitone', 'partner_f0_range_iqr_semitone', 'IQR range (semitones)'),
        ('f0_sd_semitone', 'partner_f0_sd_semitone', 'SD (semitones)'),
    ]:
        lines.append(f'\n--- Range mimicry: speaker {label} ~ partner {label} ---')
        df_m = df.dropna(subset=[metric, partner_metric])
        if len(df_m) >= 3:
            r, p_r = stats.pearsonr(df_m[metric], df_m[partner_metric])
            slope, intercept, _, p_s, se = stats.linregress(df_m[partner_metric], df_m[metric])
            lines.append(f'  Overall: n={len(df_m)}  r={r:+.3f} (p={p_r:.4f} {_sig(p_r)})  '
                          f'slope={slope:+.3f} SE={se:.3f} (p={p_s:.4f} {_sig(p_s)})')
        else:
            lines.append(f'  Overall: insufficient data (n={len(df_m)})')

        lines.append(f'  {"Group":<28} {"n":>4}  {"r":>7}  {"slope":>8}  {"p":>8}')
        lines.append('  ' + '-' * 62)
        for sg in ['Female', 'Male']:
            for pg in ['Female', 'Male']:
                sub = df_m[(df_m['gender_en'] == sg) & (df_m['partner_gender_en'] == pg)]
                if len(sub) < 5:
                    continue
                r, p_r = stats.pearsonr(sub[metric], sub[partner_metric])
                slope, _, _, p_s, _ = stats.linregress(sub[partner_metric], sub[metric])
                label_str = f'{sg} speaker -> {pg} partner'
                lines.append(f'  {label_str:<28} {len(sub):>4}  {r:>+7.3f}  {slope:>+8.3f}  {p_s:>8.4f} {_sig(p_s)}')

    lines.append('\n' + '=' * 70)
    report = '\n'.join(lines)
    logger.info(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'dynamic_range_statistics.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f'[output] Statistics saved: {output_dir / "dynamic_range_statistics.txt"}')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _raincloud(ax, sub, metric, order, palette, width=0.6):
    """
    Raincloud plot on one axis: half-violin (cloud) + slim boxplot (dot+whisker
    summary) + jittered strip (rain), offset side-by-side per category.

    Built from seaborn/matplotlib primitives (no extra dependency): the "half"
    violin is achieved by drawing a full violin then clipping away the right
    half of each patch.
    """
    sns.violinplot(data=sub, x='partner_gender_en', y=metric, order=order,
                    hue='partner_gender_en', palette=palette, ax=ax,
                    inner=None, cut=0, linewidth=0, legend=False,
                    width=width, density_norm='width')
    # Clip each violin to its left half and shift it left, opening room for
    # the strip+box on the right — the classic raincloud layout.
    for i, violin in enumerate(ax.collections):
        verts = violin.get_paths()[0].vertices
        center = i
        verts[:, 0] = np.clip(verts[:, 0], -np.inf, center)
        verts[:, 0] -= 0.02

    sns.stripplot(data=sub, x='partner_gender_en', y=metric, order=order,
                  hue='partner_gender_en', palette=palette, ax=ax,
                  size=3, alpha=0.5, jitter=0.08, legend=False,
                  native_scale=True, dodge=False,
                  zorder=1)
    # Shift the strip points right of the violin, and boxplot further right still.
    n_cats = len(order)
    for coll_idx, coll in enumerate(ax.collections[n_cats:]):
        offsets = coll.get_offsets()
        if len(offsets):
            offsets = np.array(offsets)
            offsets[:, 0] += 0.14
            coll.set_offsets(offsets)

    sns.boxplot(data=sub, x='partner_gender_en', y=metric, order=order,
                hue='partner_gender_en', palette=palette, ax=ax,
                width=0.12, showcaps=True, legend=False,
                boxprops={'zorder': 3, 'alpha': 0.8},
                whiskerprops={'zorder': 3}, medianprops={'zorder': 3, 'color': 'black'},
                flierprops={'markersize': 3, 'alpha': 0.5})
    for patch in ax.patches:
        path = patch.get_path()
        path.vertices[:, 0] += 0.30
    for line in ax.lines:
        xdata = line.get_xdata()
        line.set_xdata(np.asarray(xdata) + 0.30)


def plot_range_by_gender_pair(df, metric, y_label, title, out_path):
    """F0 range by speaker gender x partner gender (2-panel raincloud plot)."""
    order = ['Female', 'Male']
    palette = 'Set2'
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    for ax, focal_gender in zip(axes, order):
        sub = df[df['gender_en'] == focal_gender]
        if sub.empty:
            ax.set_title(f'{focal_gender} (no data)')
            continue
        _raincloud(ax, sub, metric, order, palette)
        ax.set_title(f'Focal: {focal_gender}')
        ax.set_xlabel('Partner gender')
        ax.set_ylabel(y_label)
        for x_pos, label in enumerate(order):
            n = len(sub[sub['partner_gender_en'] == label])
            ax.text(x_pos, ax.get_ylim()[0], f'n={n}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_PARTNER_COLORS = {'Female': '#e41a1c', 'Male': '#377eb8'}


def plot_range_mimicry(df, metric, partner_metric, y_label, x_label, title, out_path):
    """
    Core speaker's dynamic range vs. partner's dynamic range (same
    conversation) — faceted by speaker gender, colored by partner gender.
    Tests whether a speaker's range tracks their partner's range.
    """
    df_scatter = df.dropna(subset=[metric, partner_metric])
    if df_scatter.empty:
        return

    lo = min(df_scatter[metric].min(), df_scatter[partner_metric].min())
    hi = max(df_scatter[metric].max(), df_scatter[partner_metric].max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    for ax, sg in zip(axes, ['Female', 'Male']):
        sub_sg = df_scatter[df_scatter['gender_en'] == sg]
        slope_lines = []
        for pg, color in _PARTNER_COLORS.items():
            sub_pg = sub_sg[sub_sg['partner_gender_en'] == pg]
            if sub_pg.empty:
                continue
            sns.scatterplot(data=sub_pg, x=partner_metric, y=metric,
                            color=color, alpha=0.6, ax=ax, label=pg)
            if len(sub_pg) >= 5:
                sns.regplot(data=sub_pg, x=partner_metric, y=metric,
                            scatter=False, ax=ax, color=color,
                            line_kws={'linewidth': 2, 'linestyle': '--'})
                slope, _, r, p, _ = stats.linregress(sub_pg[partner_metric], sub_pg[metric])
                slope_lines.append(f'{pg[0]}: slope={slope:+.3f} R²={r**2:.3f} p={p:.3f} {_sig(p)}')

        ax.plot([lo, hi], [lo, hi], color='gray', linewidth=1, linestyle=':', zorder=0, label='y=x')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Focal speaker: {sg}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label if sg == 'Female' else '')
        ax.legend(title='Partner gender')
        if slope_lines:
            ax.annotate('\n'.join(slope_lines), xy=(0.03, 0.97),
                        xycoords='axes fraction', ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    speaker_meta, conv_speaker_pairs, _conv_relationship = load_metadata(CORPUS_ROOT)

    cache = load_cache(CACHE_FILE)

    def _fmin_fmax(gender):
        if gender == '女性':
            return FMIN_FEMALE, FMAX_FEMALE
        elif gender == '男性':
            return FMIN_MALE, FMAX_MALE
        return FMIN_DEFAULT, FMAX_DEFAULT

    tasks = []
    for conv_id, pair in conv_speaker_pairs.items():
        for role in ('core', 'partner'):
            spk = pair[role]
            key = (conv_id, spk['ic_label'])
            if key in cache:
                continue

            wav_path, trans_path = resolve_paths(conv_id, CORPUS_ROOT, spk['ic_label'])
            if wav_path is None:
                continue

            gender = speaker_meta.get(spk['speaker_id'], {}).get('gender')
            fmin, fmax = _fmin_fmax(gender)

            tasks.append((conv_id, str(wav_path), str(trans_path), spk['ic_label'], fmin, fmax))

    logger.info(f'[main] {len(tasks)} speaker/conversation extractions to run '
                f'({len(cache)} already cached)')

    if tasks:
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(N_WORKERS) as pool:
            new_results = pool.map(extract_f0_range_for_speaker, tasks)
        for r in new_results:
            cache[(r['conv_id'], r['ic_label'])] = r
        save_cache(cache, CACHE_FILE)

    range_results = {k: v for k, v in cache.items() if v.get('valid')}
    n_invalid = len(cache) - len(range_results)
    logger.info(f'[main] Valid extractions: {len(range_results)}  |  invalid: {n_invalid}')

    df = build_range_df(conv_speaker_pairs, speaker_meta, range_results)
    df.to_csv(RESULTS_CSV, index=False)
    logger.info(f'[output] Results saved: {RESULTS_CSV}')

    run_statistics(df, OUTPUT_DIR)

    plot_range_by_gender_pair(
        df, 'f0_range_iqr_semitone', 'F0 IQR range (semitones)',
        'F0 dynamic range (IQR) by speaker gender x partner gender',
        PLOTS_DIR / 'range_iqr_by_gender_pair.png',
    )
    plot_range_by_gender_pair(
        df, 'f0_sd_semitone', 'F0 SD (semitones)',
        'F0 dynamic range (SD) by speaker gender x partner gender',
        PLOTS_DIR / 'range_sd_by_gender_pair.png',
    )

    plot_range_mimicry(
        df, 'f0_range_iqr_semitone', 'partner_f0_range_iqr_semitone',
        'Speaker F0 IQR range (semitones)', 'Partner F0 IQR range (semitones)',
        'Range mimicry: speaker IQR range vs. partner IQR range',
        PLOTS_DIR / 'range_iqr_mimicry.png',
    )
    plot_range_mimicry(
        df, 'f0_sd_semitone', 'partner_f0_sd_semitone',
        'Speaker F0 SD (semitones)', 'Partner F0 SD (semitones)',
        'Range mimicry: speaker SD vs. partner SD',
        PLOTS_DIR / 'range_sd_mimicry.png',
    )
    logger.info(f'[output] Plots saved to {PLOTS_DIR}')


if __name__ == '__main__':
    main()
