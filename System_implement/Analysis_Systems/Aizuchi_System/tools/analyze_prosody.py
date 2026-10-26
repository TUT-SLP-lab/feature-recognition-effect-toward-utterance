"""Measure pitch (F0) and speech rate of the participant and of the system voice, per utterance.

Usage:  python tools/analyze_prosody.py logs/pilot_test/segawa_female [more sessions ...]    (a session = a folder with chatlog.jsonl)
        python tools/analyze_prosody.py --wav some.wav [--start 3.0 --end 7.5] [--fmin 70 --fmax 400]
        python tools/analyze_prosody.py logs/pilot_test/segawa_female --out logs/prosody.csv

Session mode reads mix_stereo.wav (left = participant, right = system voice) and the session log, and measures every
system clip and every participant utterance (user / backchannel) the log has. Per utterance it gives:
  f0 median / mean / sd (Hz), f0 range in semitones (5th-95th percentile), f0 sd in semitones, voiced share, and the
  F0 slope in semitones per second (a falling or rising ending is the end-of-phrase contour, last 300 ms, in semitones);
  speech rate in morae per second (mora_s) and per second of speech only (artic_mora_s, pauses > 250 ms left out). Morae are counted
  from the text: the script for system clips, the ASR transcript for participant utterances (SudachiPy reads the text as kana).
  Audio files and utterances without text have no mora rate; for them rate_syll_s / artic_syll_s give an acoustic estimate
  (syllable-like peaks in the loudness curve), which is only good for comparing voices, not an exact mora count.
Pitch is estimated with librosa's pYIN; set --fmin / --fmax for the speaker (default 65-500 Hz). Needs librosa, soundfile, numpy, scipy.
Aizuchi are left out of the printed summary: utterances the log marks as a backchannel, system clips of kind backchannel, and
participant sounds that overlap an aizuchi you marked with annotate_aizuchi.py. The CSV keeps every utterance with an "aizuchi" column.
Per-session rows are written to <session>/prosody.csv (or --out for all sessions together), and a summary by speaker is printed.
"""
import argparse
import csv
import json
import statistics as st
import threading
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

SR = 16000
HOP = 160  # 10 ms
MIN_DUR = 0.4  # shorter sounds give no reliable contour


_tok = None
_tok_lock = threading.Lock()  # the web app analyzes several conversations at once; a SudachiPy tokenizer must not be shared between threads
SMALL = set("ャュョァィゥェォヮ")  # small kana join the kana before them: one mora together


def count_mora(text: str) -> int:
    """Number of morae in Japanese text: read it with SudachiPy, then count katakana (ッ, ン and ー count one each; small ャュョ etc. none)."""
    global _tok
    with _tok_lock:
        if _tok is None:
            from sudachipy import dictionary, tokenizer
            d = dictionary.Dictionary()
            _tok = (d.tokenizer() if hasattr(d, "tokenizer") else d.create(), tokenizer.Tokenizer.SplitMode.C)
        kana = "".join(m.reading_form() or m.surface() for m in _tok[0].tokenize(text, _tok[1]))
    kana = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kana)  # hiragana -> katakana
    return sum(1 for c in kana if ("ァ" <= c <= "ヺ" or c == "ー") and c not in SMALL)  # digits and Latin letters have no reading here and are skipped


def semitones(f, ref):
    return 12 * np.log2(f / ref)


def analyze(x: np.ndarray, sr: int, fmin=65.0, fmax=500.0, contour=False) -> dict | None:
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    dur = len(x) / SR
    if dur < MIN_DUR:
        return None
    f0, voiced, _ = librosa.pyin(x, fmin=fmin, fmax=fmax, sr=SR, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=x, frame_length=400, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-6)
    n = min(len(f0), len(db))
    f0, voiced, db = f0[:n], voiced[:n], db[:n]
    out = {"dur_s": round(dur, 2), "voiced_share": round(float(voiced.mean()), 2)}
    if contour:  # [seconds, Hz] every 30 ms, None where unvoiced (for plotting)
        out["contour"] = [[round(i * 0.01, 2), round(float(f0[i]), 1) if voiced[i] and not np.isnan(f0[i]) else None] for i in range(0, n, 3)]
    v = f0[voiced & ~np.isnan(f0)]
    if len(v) >= 10:
        ref = float(np.median(v))
        s = semitones(v, ref)
        out.update(f0_median=round(ref, 1), f0_mean=round(float(v.mean()), 1), f0_sd=round(float(v.std()), 1),
                   f0_range_st=round(float(np.percentile(s, 95) - np.percentile(s, 5)), 2), f0_sd_st=round(float(s.std()), 2))
        tail = np.where(voiced[-30:] & ~np.isnan(f0[-30:]))[0]  # last 300 ms: ending contour
        if len(tail) >= 5:
            t = (tail) * 0.01
            out["end_slope_st_s"] = round(float(np.polyfit(t, semitones(f0[-30:][tail], ref), 1)[0]), 1)
    # syllable-like peaks: loudness peaks that rise >= 2 dB over the dip before them, only where there is voicing nearby
    sm = np.convolve(db, np.ones(5) / 5, mode="same")
    floor = np.percentile(sm, 95) - 30
    near_voice = np.convolve(voiced.astype(float), np.ones(7), mode="same") > 0
    pk, _ = find_peaks(sm, prominence=2.0, distance=8, height=floor)
    pk = [p for p in pk if near_voice[p]]
    silent = sm < np.percentile(sm, 95) - 25
    pause = 0.0  # total of pauses longer than 250 ms, edges included only if inside the sound
    run = 0
    for s in list(silent) + [False]:
        if s:
            run += 1
        else:
            if run * 0.01 > 0.25:
                pause += run * 0.01
            run = 0
    speech = max(dur - pause, 0.1)
    out.update(syll=len(pk), pause_s=round(pause, 2), rate_syll_s=round(len(pk) / dur, 2), artic_syll_s=round(len(pk) / speech, 2))
    return out


def session_rows(d: Path, fmin, fmax, contour=False) -> list[dict]:
    import annotate_aizuchi as A  # reuse the session reader (makes the mixes if they are missing)
    info = A.session_info(str(d.resolve().relative_to(A.LOGS)))
    x, sr = sf.read(d / "mix_stereo.wav", dtype="float32")
    rows = []
    f = d / "annotations.json"  # aizuchi you marked by ear (annotate_aizuchi.py) also count as aizuchi here
    marks = [m["t"] for m in json.loads(f.read_text(encoding="utf-8")).get("marks", []) if m["type"] == "aizuchi"] if f.exists() else []
    jobs = [("system", c["start"], c["end"], c.get("kind") or "", c.get("text", "")) for c in info["clips"]]
    jobs += [(e["role"], e["start"], e["end"], "", e["text"]) for e in info["events"]]
    for who, a, b, kind, text in sorted(jobs, key=lambda j: j[1]):
        ch = x[int(a * sr):int(b * sr), 1 if who == "system" else 0]
        r = analyze(ch, sr, fmin, fmax, contour)
        if r:
            if contour:
                r["contour"] = [[round(t + a, 2), h] for t, h in r["contour"]]  # absolute time in the recording
            is_aiz = who == "backchannel" or kind in ("backchannel", "aizuchi") or (who != "system" and any(a - 0.3 <= t <= b + 0.3 for t in marks))
            r.update(session=str(d.name), speaker="system" if who == "system" else "participant", kind=kind or who, start_s=round(a, 2), text=text,
                     aizuchi="yes" if is_aiz else "")
            m = count_mora(text) if text else 0
            if m:  # morae from the text (system script / ASR transcript) over the clip length
                r.update(mora=m, mora_s=round(m / r["dur_s"], 2), artic_mora_s=round(m / max(r["dur_s"] - r["pause_s"], 0.1), 2))
            rows.append(r)
    return rows


def split_utterances(x: np.ndarray, sr: int, top_db=30, gap_s=0.3):
    """Cut one recording into stretches of sound separated by pauses longer than gap_s -> [(start_s, end_s), ...]."""
    iv = librosa.effects.split(x, top_db=top_db, frame_length=1024, hop_length=256)
    out = []
    for a, b in iv:
        if out and (a - out[-1][1]) / sr < gap_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a / sr, b / sr) for a, b in out if (b - a) / sr >= MIN_DUR]


def summary(rows, include_aizuchi=False):
    if not include_aizuchi:
        n = sum(1 for r in rows if r.get("aizuchi"))
        rows = [r for r in rows if not r.get("aizuchi")]
        print(f"\n(summary leaves out {n} aizuchi / backchannel utterances; use --include-aizuchi to keep them)")
    for who in ("participant", "system"):
        rs = [r for r in rows if r["speaker"] == who]
        if not rs:
            continue
        print(f"\n{who}: {len(rs)} utterances")
        for k in ("f0_median", "f0_range_st", "f0_sd_st", "end_slope_st_s", "mora_s", "artic_mora_s", "rate_syll_s", "artic_syll_s"):
            vs = [r[k] for r in rs if r.get(k) is not None]
            if vs:
                print(f"  {k:15s} median {st.median(vs):7.2f}   mean {st.mean(vs):7.2f}   sd {st.pstdev(vs):6.2f}   n={len(vs)}")


FIELDS = ["session", "speaker", "kind", "aizuchi", "start_s", "dur_s", "f0_median", "f0_mean", "f0_sd", "f0_range_st", "f0_sd_st", "end_slope_st_s",
          "voiced_share", "syll", "pause_s", "mora", "mora_s", "artic_mora_s", "rate_syll_s", "artic_syll_s", "text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--wav")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--fmin", type=float, default=65.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--out")
    ap.add_argument("--include-aizuchi", action="store_true", help="keep aizuchi / backchannel utterances in the printed summary")
    a = ap.parse_args()
    if a.wav:
        x, sr = sf.read(a.wav, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = x[int(a.start * sr):int(a.end * sr) if a.end else None]
        print(analyze(x, sr, a.fmin, a.fmax) or "too short to analyze")
        return
    if not a.sessions:
        ap.error("give a session folder or --wav")
    allrows = []
    for s in a.sessions:
        d = Path(s)
        rows = session_rows(d, a.fmin, a.fmax)
        allrows += rows
        if not a.out:
            with open(d / "prosody.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"{d}: {len(rows)} utterances -> {d / 'prosody.csv'}")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(allrows)
        print(f"{len(allrows)} utterances -> {a.out}")
    summary(allrows, a.include_aizuchi)


if __name__ == "__main__":
    main()
