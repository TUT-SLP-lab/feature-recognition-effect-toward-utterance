"""Listen to a session and mark where the participant gives an aizuchi (or another reaction), to find the ones the ASR missed.

Usage:  python tools/annotate_aizuchi.py            (then open http://localhost:8001)
        PORT=8002 HOST=0.0.0.0 python tools/annotate_aizuchi.py

It needs no models and no GPU. It reads the session folders under logs/ (each with chatlog.jsonl and audio/), makes the
merged mixes if they are missing (see merge_session_audio.py / rebalance_mix.py), and shows the participant and the
system voice as two lanes with the system's cues (|S / |W), what the system detected, and suggested sounds the
detector found. Marks are saved after every change to <session>/annotations.json and annotations.csv.
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from merge_session_audio import merge  # noqa: E402
from rebalance_mix import rebalance  # noqa: E402

LOGS = (HERE.parent / "logs").resolve()
STATIC = HERE / "annotate_static"
PEAK_S = 0.02  # waveform resolution: one peak per 20 ms
app = FastAPI()
_cache: dict = {}


def session_dir(rel: str) -> Path:
    d = (LOGS / rel).resolve()
    if LOGS not in d.parents or not (d / "chatlog.jsonl").is_file():
        raise HTTPException(404, "no such session")
    return d


def ensure_mixes(d: Path):
    if not (d / "mix_stereo.wav").exists():
        merge(d)
    if not (d / "mix_mono_balanced.wav").exists():
        rebalance(d, 0.12, 0.07)


def read_rows(d: Path) -> list[dict]:
    return [json.loads(line) for line in (d / "chatlog.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def peaks(x: np.ndarray, sr: int) -> list[int]:
    w = int(PEAK_S * sr)
    n = len(x) // w
    p = np.abs(x[: n * w]).reshape(n, w).max(1)
    return p.tolist()


def find_candidates(human: np.ndarray, sr: int) -> list[dict]:
    """Short sounds in the participant channel (energy bursts), whether or not the system noticed them."""
    w = int(0.03 * sr)
    lvl = np.convolve(np.abs(human), np.ones(w) / w, mode="same")
    th = max(0.008, float(np.percentile(lvl, 50)) * 20)
    on = lvl > th
    gap, out, i, n = int(0.25 * sr), [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or on[j: j + gap].any()):
                j += 1
            if (j - i) / sr >= 0.08:
                out.append({"start": round(i / sr, 3), "end": round(j / sr, 3)})
            i = j
        else:
            i += 1
    return out


def session_info(rel: str) -> dict:
    d = session_dir(rel)
    ensure_mixes(d)
    key = (rel, (d / "mix_stereo.wav").stat().st_mtime, (d / "chatlog.jsonl").stat().st_mtime)
    if key in _cache:
        return _cache[key]
    x, sr = sf.read(d / "mix_stereo.wav", dtype="float32")
    human, system = x[:, 0], x[:, 1]
    rows = read_rows(d)
    ai = {r["audio"]: r for r in rows if r["role"] == "ai" and r.get("audio")}
    clips = []
    for r in rows:
        if r["role"] == "event" and r["text"].startswith("ai played") and r.get("clip") and r.get("full_offset_s") is not None:
            a = ai.get(r["clip"], {})
            clips.append({"start": r["full_offset_s"], "end": r["end_s"], "cue": a.get("cue"), "kind": a.get("script_kind") or a.get("source"),
                          "source": a.get("source"), "text": a.get("text", "")})
    events = []
    for r in rows:
        if r["role"] in ("user", "backchannel") and r.get("full_offset_s") is not None:
            start = r["full_offset_s"] + 0.1  # the saved clip starts about 0.1 s before the sound
            events.append({"start": round(start, 3), "end": round(start + max(0.3, r["duration_s"] - 0.9), 3), "role": r["role"],
                           "text": r["text"], "reclassified": bool(r.get("reclassified"))})
    cands = find_candidates(human, sr)
    for c in cands:  # what the system already knows about this sound
        hit = next((e for e in events if c["start"] < e["end"] + 0.3 and c["end"] > e["start"] - 0.3), None)
        c["logged"] = hit["role"] if hit else None
    hp = peaks(human, sr)
    ref = float(np.percentile(hp, 99.5)) or 1.0
    info = {"path": rel, "duration": round(len(human) / sr, 3), "peak_s": PEAK_S,
            "human": [min(100, round(v / ref * 100)) for v in hp], "system": [round(v * 100) for v in peaks(system, sr)],
            "clips": clips, "events": events, "candidates": cands,
            "audio": f"/audio/{rel}/mix_mono_balanced.wav", "script": next((r["text"] for r in rows if r["role"] == "event" and "script started" in r["text"]), None)}
    _cache[key] = info
    return info


@app.get("/api/sessions")
def api_sessions():
    out = []
    for f in sorted(LOGS.glob("**/chatlog.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        d = f.parent
        if not (d / "audio").is_dir() or not list((d / "audio").glob("human_full_*.wav")):
            continue
        a = d / "annotations.json"
        n = len(json.loads(a.read_text(encoding="utf-8")).get("marks", [])) if a.exists() else None
        script = voice = None
        for r in read_rows(d):  # what this run used (logged since the settings snapshot was added; older sessions show nothing)
            if script is None and r["role"] == "event" and r["text"].startswith("script started"):
                script = r.get("script_file") or r["text"].split(":", 1)[-1].strip()
            if voice is None and r["role"] == "ai" and r.get("tts_model"):
                voice = r["tts_model"]
            if script and voice:
                break
        out.append({"path": d.relative_to(LOGS).as_posix(), "marks": n, "script": script, "voice": voice})
    return out


@app.get("/api/session")
def api_session(path: str):
    info = dict(session_info(path))
    a = session_dir(path) / "annotations.json"
    info["marks"] = json.loads(a.read_text(encoding="utf-8")).get("marks", []) if a.exists() else []
    return info


@app.get("/audio/{rel:path}")
def audio(rel: str):
    p = (LOGS / rel).resolve()
    if LOGS not in p.parents or p.suffix != ".wav" or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")  # Starlette answers Range requests, so seeking works


def describe(mark: dict, info: dict) -> dict:
    """What the system saw at the time of a mark: a logged event (did the ASR / detector catch it) and where the mark falls
    relative to the system's clips (inside a clip, or after one), plus the signed offset to the nearest clip end."""
    t = mark["t"]
    ev = next((e for e in info["events"] if e["start"] - 0.5 <= t <= e["end"] + 0.5), None)
    out = {"logged_role": ev["role"] if ev else None, "logged_text": ev["text"] if ev else None,
           "caught_by_system": bool(ev and ev["role"] == "backchannel")}
    clips = info["clips"]
    if not clips:
        return out
    inside = next((c for c in clips if c["start"] <= t <= c["end"]), None)
    prev = [c for c in clips if c["end"] < t]
    ref = inside or (max(prev, key=lambda c: c["end"]) if prev else None)
    near = min(clips, key=lambda c: abs(c["end"] - t))
    out.update(during_clip=bool(inside), nearest_clip_cue=near["cue"] or near["kind"], offset_from_nearest_clip_end=round(t - near["end"], 3))
    if ref:
        out.update(system_clip_cue=ref["cue"], system_clip_kind=ref["kind"], system_clip_text=ref["text"],
                   seconds_into_clip=round(t - ref["start"], 3) if inside else None, seconds_after_clip_end=round(t - ref["end"], 3) if not inside else None)
    return out


@app.post("/api/annotations")
async def api_annotations(path: str, request: Request):
    d = session_dir(path)
    body = await request.json()
    info = session_info(path)
    marks = []
    for m in body.get("marks", []):
        m = {k: m[k] for k in ("id", "t", "type", "note") if k in m}
        m["t"] = round(float(m["t"]), 3)
        m.update(describe(m, info))
        marks.append(m)
    marks.sort(key=lambda m: m["t"])
    (d / "annotations.json").write_text(json.dumps({"session": path, "saved_at": datetime.now().isoformat(timespec="seconds"), "marks": marks},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["t", "type", "note", "caught_by_system", "logged_role", "logged_text", "during_clip", "nearest_clip_cue", "offset_from_nearest_clip_end",
            "system_clip_cue", "system_clip_kind", "seconds_into_clip", "seconds_after_clip_end", "system_clip_text"]
    with open(d / "annotations.csv", "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: opens correctly in Excel
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(marks)
    return {"ok": True, "marks": marks}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    # timeout_graceful_shutdown: Ctrl+C used to hang while the browser still held an audio stream open (half-closed connections)
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8001)), timeout_graceful_shutdown=2)
