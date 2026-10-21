"""Make the participant louder and the system voice quieter in an already merged session (mix_stereo.wav).

Usage:  python tools/rebalance_mix.py logs/pilot_test/segawa_male [more folders ...] [--human 0.12] [--system 0.07]

Reads <session>/mix_stereo.wav (left = participant, right = system voice, made by merge_session_audio.py) and writes
  mix_stereo_balanced.wav   the two channels with the new levels
  mix_mono_balanced.wav     both summed
The originals (audio/human_full_01.wav, audio/ai_full_01.wav) and the unbalanced mixes are never modified.
--human / --system are the target loudness (RMS over the parts where that voice is active; 0.1 is about normal speech level).
A soft limiter keeps loud answers from clipping after the boost.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def active_rms(x, sr, thr=0.01):
    w = int(0.05 * sr)
    fr = x[: len(x) // w * w].reshape(-1, w)
    r = np.sqrt((fr ** 2).mean(1))
    a = r[r > thr]
    return float(np.sqrt((a ** 2).mean())) if len(a) else 0.0


def soft_limit(x, knee=0.7):
    ax = np.abs(x)
    return np.where(ax <= knee, x, np.sign(x) * (knee + (1 - knee) * np.tanh((ax - knee) / (1 - knee))))


def rebalance(session: Path, human_target: float, system_target: float):
    x, sr = sf.read(session / "mix_stereo.wav", dtype="float32")
    gh = human_target / max(active_rms(x[:, 0], sr), 1e-6)
    gs = system_target / max(active_rms(x[:, 1], sr), 1e-6)
    h, a = soft_limit(x[:, 0] * gh), soft_limit(x[:, 1] * gs)
    sf.write(session / "mix_stereo_balanced.wav", np.stack([h, a], axis=1), sr, subtype="PCM_16")
    mono = soft_limit(h + a, 0.8)
    sf.write(session / "mix_mono_balanced.wav", mono, sr, subtype="PCM_16")
    print(f"{session.name}: participant x{gh:.2f}, system x{gs:.2f}; peaks L {np.abs(h).max():.2f} R {np.abs(a).max():.2f} mono {np.abs(mono).max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--human", type=float, default=0.12)
    ap.add_argument("--system", type=float, default=0.07)
    args = ap.parse_args()
    for s in args.sessions:
        rebalance(Path(s), args.human, args.system)
