"""Merge one session's participant and system recordings into files for listening and marking.

Usage:  python tools/merge_session_audio.py logs/pilot_test/segawa_male [more session folders ...]

Writes into each session folder:
  mix_stereo.wav     left = participant (mic), right = system voice, same timeline (open in Audacity to see both)
  mix_mono.wav       both voices summed (participant boosted to be audible next to the system voice), for any player
  labels_system.txt  Audacity labels: every system clip that was played (cue S/W, kind, text)
  labels_user.txt    Audacity labels: every sound the page detected from the participant (backchannel / user, with the transcript)
Import a label file in Audacity with File > Import > Labels, then add your own label track to mark aizuchi and reactions.
"""
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: Path):
    x, sr = sf.read(path, dtype="float32")
    return (x if x.ndim == 1 else x[:, 0]), sr


def merge(session: Path):
    human, hsr = load_mono(session / "audio" / "human_full_01.wav")
    ai, asr_ = load_mono(session / "audio" / "ai_full_01.wav")
    if asr_ != hsr:  # the two tracks are on the same timeline but were recorded at different sample rates
        g = gcd(hsr, asr_)
        ai = resample_poly(ai, hsr // g, asr_ // g).astype(np.float32)
    n = max(len(human), len(ai))
    human, ai = np.pad(human, (0, n - len(human))), np.pad(ai, (0, n - len(ai)))
    sf.write(session / "mix_stereo.wav", np.stack([human, ai], axis=1), hsr, subtype="PCM_16")
    # The mic is usually much quieter than the system voice: in the mono file the participant is boosted (up to x6) so that
    # a quiet "un" can be heard next to the system voice. mix_stereo.wav keeps both channels untouched.
    p_h, p_a = float(np.percentile(np.abs(human), 99.5)), float(np.percentile(np.abs(ai), 99.5))
    gain = min(6.0, max(1.0, p_a / p_h)) if p_h > 0 else 1.0
    mono = human * gain + ai
    peak = float(np.abs(mono).max())
    sf.write(session / "mix_mono.wav", mono / peak * 0.95 if peak > 0.95 else mono, hsr, subtype="PCM_16")

    rows = [json.loads(line) for line in (session / "chatlog.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ai_rows = {r["audio"]: r for r in rows if r["role"] == "ai" and r.get("audio")}
    sysl, usel = [], []
    for r in rows:
        if r["role"] == "event" and r["text"].startswith("ai played") and r.get("clip") and r.get("full_offset_s") is not None:
            a = ai_rows.get(r["clip"], {})
            tag = a.get("cue") or a.get("script_kind") or a.get("source") or "?"
            sysl.append((r["full_offset_s"], r["end_s"], f"[{tag}] {a.get('text', '')}"))
        elif r["role"] in ("user", "backchannel") and r.get("full_offset_s") is not None:
            start = r["full_offset_s"]
            end = start + max(0.3, r["duration_s"] - 1.0)  # the saved clip ends with about 1 s of waiting for you to finish
            usel.append((start, end, f"[{r['role']}] {r['text']}"))
    for name, items in (("labels_system.txt", sysl), ("labels_user.txt", usel)):
        (session / name).write_text("".join(f"{a:.3f}\t{b:.3f}\t{t}\n" for a, b, t in sorted(items)), encoding="utf-8")
    print(f"{session.name}: {n / hsr:.1f} s, {hsr} Hz, participant boosted x{gain:.1f} in mix_mono, {len(sysl)} system labels, {len(usel)} participant labels")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        merge(Path(arg))
