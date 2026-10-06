#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFP batch consistency test.

Runs the same age/gender pairing N times through Response_All_AI_weight_random's
--mode debug generation (dice roll + enforcement retry + best-of-N ranking, same
as a normal debug run), then aggregates every run's *_sfp_debug.csv into averaged
statistics: rolled distribution (the dice), used distribution (the actual output),
and category/tier match rate (the "shift"/drift away from what was rolled).

Each run is saved under its own timestamped folder exactly like a normal --mode
debug invocation (via Response_All_AI_weight_random.run_chat), so nothing about
a single run's output format changes — this script only adds the aggregation
step on top of N of them.

Requires GENAI_API_KEY to be set, same as the main script.

Usage:
    python3 SFP_batch_test.py --runs 5 --turns 100
    python3 SFP_batch_test.py --runs 10 --turns 50 --age-a 20s --gender-a male --age-b 20s --gender-b female
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

import Response_All_AI_weight_random as core

AGE_BUCKETS = core.AGE_BUCKETS
SFP_TIER_LIST = core.SFP_TIER_LIST
BATCH_OUTPUT_DIR = core.OUTPUTS_DIR / "sfp_batch_test"


def run_one(profile_a, profile_b, turns, out_dir):
    """One full debug-mode generation (dice roll + retries + ranking), saved under
    out_dir exactly like a normal --mode debug run. Returns the debug CSV path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{profile_a['id']}_vs_{profile_b['id']}.csv"
    core.run_chat(profile_a, profile_b, turns, output_path, debug_sfp=True)
    return output_path.with_name(f"{output_path.stem}_sfp_debug.csv")


def per_run_stats(debug_csv_path):
    """One run's debug CSV -> per-speaker stats: rolled_category %, used_category %,
    rolled_tier %, used_tier %, category_match_rate, tier_match_rate, mean
    attempts_used. Returns a dict keyed by speaker_id."""
    df = pd.read_csv(debug_csv_path)
    stats = {}
    for speaker_id, sub in df.groupby('speaker_id'):
        stats[speaker_id] = {
            'n': len(sub),
            'rolled_category_pct': sub['rolled_category'].value_counts(normalize=True) * 100,
            'used_category_pct': sub['used_category'].value_counts(normalize=True) * 100,
            'rolled_tier_pct': sub['rolled_tier'].value_counts(normalize=True) * 100,
            'used_tier_pct': sub['used_tier'].value_counts(normalize=True) * 100,
            'category_match_rate': (sub['used_category'] == sub['rolled_category']).mean() * 100,
            'tier_match_rate': (sub['used_tier'] == sub['rolled_tier']).mean() * 100,
            'mean_attempts_used': sub['attempts_used'].mean(),
        }
    return stats


CATEGORY_LABELS = ['feminine', 'neutral', 'masculine']
USED_CATEGORY_LABELS = ['feminine', 'neutral', 'masculine', 'no_sfp']


def average_series(series_list, labels):
    """Average a list of pandas Series (one per run, index = category/tier labels,
    values = %) into one Series over the given fixed label set, filling 0.0 for a
    label a given run never produced — so one run's silence on a rare tier doesn't
    skip it from the average, it just counts as 0% for that run."""
    aligned = [s.reindex(labels, fill_value=0.0) for s in series_list]
    return pd.concat(aligned, axis=1).mean(axis=1)


def aggregate(all_run_stats):
    """List of per-run stats dicts (one per completed run) -> one averaged summary
    per speaker_id. Each run contributes one data point per metric (not raw rows),
    so a long run can't outweigh a short one."""
    speaker_ids = sorted({sid for run in all_run_stats for sid in run})
    summary = {}
    for sid in speaker_ids:
        runs_for_speaker = [run[sid] for run in all_run_stats if sid in run]
        n_runs = len(runs_for_speaker)
        summary[sid] = {
            'n_runs': n_runs,
            'mean_n_per_run': sum(r['n'] for r in runs_for_speaker) / n_runs,
            'rolled_category_pct': average_series([r['rolled_category_pct'] for r in runs_for_speaker], CATEGORY_LABELS),
            'used_category_pct': average_series([r['used_category_pct'] for r in runs_for_speaker], USED_CATEGORY_LABELS),
            'rolled_tier_pct': average_series([r['rolled_tier_pct'] for r in runs_for_speaker], SFP_TIER_LIST),
            'used_tier_pct': average_series([r['used_tier_pct'] for r in runs_for_speaker], SFP_TIER_LIST + ['no_sfp']),
            'category_match_rate_mean': sum(r['category_match_rate'] for r in runs_for_speaker) / n_runs,
            'category_match_rate_std': pd.Series([r['category_match_rate'] for r in runs_for_speaker]).std(),
            'tier_match_rate_mean': sum(r['tier_match_rate'] for r in runs_for_speaker) / n_runs,
            'tier_match_rate_std': pd.Series([r['tier_match_rate'] for r in runs_for_speaker]).std(),
            'mean_attempts_used': sum(r['mean_attempts_used'] for r in runs_for_speaker) / n_runs,
        }
    return summary


def print_summary(summary, n_runs_requested, n_runs_completed):
    print("=" * 70)
    print(f"SFP BATCH TEST SUMMARY — {n_runs_completed}/{n_runs_requested} runs completed")
    print("=" * 70)
    for sid, s in summary.items():
        print(f"\n--- {sid} (avg {s['mean_n_per_run']:.0f} turns/run, {s['n_runs']} runs) ---")
        print("Rolled category % (the dice):")
        print(s['rolled_category_pct'].round(1).to_string())
        print("\nUsed category % (the actual output):")
        print(s['used_category_pct'].round(1).to_string())
        print(f"\nCategory match rate: {s['category_match_rate_mean']:.1f}% (std {s['category_match_rate_std']:.1f})")
        print(f"Tier match rate:     {s['tier_match_rate_mean']:.1f}% (std {s['tier_match_rate_std']:.1f})")
        print(f"Mean attempts used:  {s['mean_attempts_used']:.2f}")
        print("\nRolled tier % (the dice, full detail):")
        print(s['rolled_tier_pct'].round(1).to_string())
        print("\nUsed tier % (the actual output, full detail):")
        print(s['used_tier_pct'].round(1).to_string())


def save_summary_csv(summary, out_path):
    rows = []
    for sid, s in summary.items():
        row = {
            'speaker_id': sid,
            'n_runs': s['n_runs'],
            'mean_n_per_run': round(s['mean_n_per_run'], 1),
            'category_match_rate_mean': round(s['category_match_rate_mean'], 2),
            'category_match_rate_std': round(s['category_match_rate_std'], 2),
            'tier_match_rate_mean': round(s['tier_match_rate_mean'], 2),
            'tier_match_rate_std': round(s['tier_match_rate_std'], 2),
            'mean_attempts_used': round(s['mean_attempts_used'], 3),
        }
        for label, val in s['rolled_category_pct'].items():
            row[f'rolled_category_{label}_pct'] = round(val, 2)
        for label, val in s['used_category_pct'].items():
            row[f'used_category_{label}_pct'] = round(val, 2)
        for label, val in s['rolled_tier_pct'].items():
            row[f'rolled_tier_{label}_pct'] = round(val, 2)
        for label, val in s['used_tier_pct'].items():
            row[f'used_tier_{label}_pct'] = round(val, 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved summary CSV to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the same age/gender pairing N times and report averaged SFP "
                     "dice-roll vs actual-output statistics (per-run averages, not pooled rows)."
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs (default: 5)")
    parser.add_argument("--turns", type=int, default=100, help="Turns per run (default: 100)")
    parser.add_argument("--age-a", choices=AGE_BUCKETS, default="20s")
    parser.add_argument("--gender-a", choices=["male", "female"], default="male")
    parser.add_argument("--age-b", choices=AGE_BUCKETS, default="20s")
    parser.add_argument("--gender-b", choices=["male", "female"], default="female")
    args = parser.parse_args()

    gender_category = {"male": "masculine", "female": "feminine"}
    profile_a = core.make_profile(args.age_a, args.gender_a, gender_category[args.gender_a])
    profile_b = core.make_profile(args.age_b, args.gender_b, gender_category[args.gender_b])

    batch_dir = BATCH_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Batch test: {args.runs} runs x {args.turns} turns, "
          f"{profile_a['gender_age']} vs {profile_b['gender_age']}")
    print(f"Output: {batch_dir}")

    all_run_stats = []
    for run_idx in range(1, args.runs + 1):
        print(f"\n[{run_idx}/{args.runs}] running...")
        run_dir = batch_dir / f"run_{run_idx}"
        try:
            debug_csv_path = run_one(profile_a, profile_b, args.turns, run_dir)
            all_run_stats.append(per_run_stats(debug_csv_path))
        except Exception as e:
            print(f"  Run {run_idx} failed: {e}")

    if not all_run_stats:
        print("\nNo runs completed successfully — nothing to summarize.")
        return

    summary = aggregate(all_run_stats)
    print_summary(summary, args.runs, len(all_run_stats))
    save_summary_csv(summary, batch_dir / "batch_summary.csv")


if __name__ == '__main__':
    main()
