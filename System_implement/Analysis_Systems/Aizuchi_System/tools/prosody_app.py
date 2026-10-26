"""Web page for the pitch / speech-rate analysis (analyze_prosody.py).

Usage:  python tools/prosody_app.py            (then open http://localhost:8003)
        PORT=8004 HOST=0.0.0.0 python tools/prosody_app.py

Pick 2 to 4 recorded sessions (or audio files) and compare them side by side: a summary table, the spread of each measure,
pitch curves on a shared scale, and an utterance table you can sort, play from, and download as CSV.
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_prosody as P  # noqa: E402
import annotate_aizuchi as A  # noqa: E402  (session reader)

STATIC = HERE / "prosody_static"
app = FastAPI()
_cache: dict = {}


@app.get("/api/sessions")
def api_sessions():
    out = []
    for f in sorted(A.LOGS.glob("**/chatlog.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append(str(f.parent.relative_to(A.LOGS)))
    return out


@app.post("/api/analyze_session")
def analyze_session(path: str, fmin: float = 65, fmax: float = 500):
    d = A.session_dir(path)
    A.ensure_mixes(d)
    key = (path, fmin, fmax, (d / "mix_stereo.wav").stat().st_mtime, (d / "chatlog.jsonl").stat().st_mtime)
    if key not in _cache:
        _cache[key] = P.session_rows(d, fmin, fmax, contour=True)
    return {"rows": _cache[key], "audio": f"/audio/{path}/mix_mono_balanced.wav"}


@app.post("/api/analyze_file")
async def analyze_file(file: UploadFile = File(...), fmin: float = Form(65), fmax: float = Form(500)):
    data = await file.read()
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        # formats soundfile cannot read (m4a, mp3 on old builds, ...): let librosa/ffmpeg try
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.wav").suffix) as t:
            t.write(data)
            t.flush()
            try:
                y, sr = P.librosa.load(t.name, sr=None, mono=False)
            except Exception as e:
                raise HTTPException(400, f"cannot read this audio file: {e}")
        x = np.atleast_2d(y).T
    chans = [("channel 1" if x.shape[1] > 1 else "speaker", x[:, 0])] + ([("channel 2", x[:, 1])] if x.shape[1] > 1 else [])
    rows = []
    for name, ch in chans:
        for a, b in P.split_utterances(ch, sr):
            r = P.analyze(ch[int(a * sr):int(b * sr)], sr, fmin, fmax, contour=True)
            if r:
                r["contour"] = [[round(t + a, 2), h] for t, h in r["contour"]]
                r.update(session=file.filename, speaker=name, kind="", start_s=round(a, 2), text="")
                rows.append(r)
    return {"rows": rows, "audio": None, "duration": round(len(x) / sr, 2)}


@app.get("/audio/{rel:path}")
def audio(rel: str):
    f = (A.LOGS / rel).resolve()
    if A.LOGS not in f.parents or f.suffix != ".wav" or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8003)), timeout_graceful_shutdown=2)
