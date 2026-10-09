"""
Simple cascaded voice chat: mic -> anime-whisper (ASR) -> Gemini 3.5 Flash Lite/Genma 4 12B (LLM) -> Style-Bert-VITS2 (TTS).

Run:  GENAI_API_KEY=... python simple_system.py   (then open http://localhost:8000)
LLMs: Gemini 3.5 Flash-Lite (needs GENAI_API_KEY) and local Gemma 4 12B (QAT 4-bit, served by llama.cpp); pick one in the web UI
      before starting the conversation. A model that can't load is simply greyed out.
Uses the same dependencies as ../Voice_Chat_App/requirements.txt.
"""

import asyncio
import atexit
import base64
import io
import json
import os
import queue
import random
import struct
import subprocess
import traceback
import urllib.request
import sys
import re
import zlib
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
from transformers import pipeline

SBV2_DIR = Path(__file__).resolve().parent.parent / "Style-Bert-VITS2"
sys.path.insert(0, str(SBV2_DIR))  # use the local Style-Bert-VITS2 checkout

from style_bert_vits2.constants import Languages  # noqa: E402
from style_bert_vits2.nlp import bert_models  # noqa: E402
from style_bert_vits2.tts_model import TTSModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_MODEL = "litagin/anime-whisper"  # was kotoba-tech/kotoba-whisper-v2.0 (swap back here and in ASR_KWARGS)
# anime-whisper's model card: no initial prompt, and no repetition limits (they cut off short repeated words like うんうん)
ASR_KWARGS = {"language": "ja", "task": "transcribe", "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMMA_MODEL = "google/gemma-4-12B-it-qat-q4_0-gguf"  # official quantization-aware-trained 4-bit GGUF (llama.cpp)
GEMMA_FILE = "gemma-4-12b-it-qat-q4_0.gguf"
GEMMA_MAX_NEW_TOKENS = 120  # replies are meant to be 1-2 short sentences
GEMMA_HISTORY_MESSAGES = 24  # only the most recent messages go to the local model (its context is limited)
# llama.cpp's server is started by this script (build it once: see ~/llama.cpp). Override with env vars if needed.
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", str(Path.home() / "llama.cpp/build/bin/llama-server"))
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", 8081))
LLAMA_CTX = 8192
BERT_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"
HERE = Path(__file__).resolve().parent
MODEL_ASSETS = SBV2_DIR / "model_assets"

SYSTEM_PROMPT = """\
あなたは「あおい」、25歳の女性です。
あなたは自分の前の海外旅行での思い出について話しています。
返事は話し言葉で、2〜3個の短い句をつなげて答えてください。
句の終わりには必ず読点「、」を付け、「〜ね」「〜さ」「〜て」「〜けど」のどれかで終えてください。最後の句だけ「。」で終えます。
最後の句も「〜ね。」「〜よね。」のように終え、まとめや締めくくりは言わず、続きがありそうな言い方にしてください。「〜な。」「〜だ。」では終えないでください。
例：「駅前のカフェでね、朝ごはんを食べてたらさ、隣の人が話しかけてくれたんだよね。」
質問を少なくし、自分のことも話してください。
相手が話した言葉を繰り返さないようにしてください。
"""

# ---------------- load models once at startup ----------------
asr = pipeline(
    "automatic-speech-recognition", model=ASR_MODEL, device=DEVICE, chunk_length_s=30,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

bert_models.load_model(Languages.JP, BERT_ID).float()  # SBV2 expects fp32
bert_models.load_tokenizer(Languages.JP, BERT_ID)


def list_tts_models() -> list[dict]:
    """One entry per .safetensors under model_assets/<dir>/ that has config.json + style_vectors.npy."""
    models = []
    for d in sorted(p for p in MODEL_ASSETS.iterdir() if p.is_dir()):
        if not ((d / "config.json").exists() and (d / "style_vectors.npy").exists()):
            continue
        styles = list(json.loads((d / "config.json").read_text(encoding="utf-8"))["data"].get("style2id", {"Neutral": 0}))
        for ckpt in sorted(d.glob("*.safetensors")):
            models.append({"id": f"{d.name}/{ckpt.name}", "name": d.name, "checkpoint": ckpt.name, "styles": styles})
    return models


_tts_cache: dict[str, TTSModel] = {}


def get_tts(model_id: str) -> TTSModel:
    if model_id not in _tts_cache:
        if model_id not in {m["id"] for m in list_tts_models()}:
            raise HTTPException(404, f"Unknown TTS model: {model_id}")
        folder, _, ckpt = model_id.partition("/")
        d = MODEL_ASSETS / folder
        if len(_tts_cache) >= 2:  # keep GPU memory bounded
            del _tts_cache[next(iter(_tts_cache))]
        _tts_cache[model_id] = TTSModel(
            model_path=d / ckpt, config_path=d / "config.json", style_vec_path=d / "style_vectors.npy", device=DEVICE,
        )
    return _tts_cache[model_id]


# ---------------- LLM backends ----------------
# Conversation = list of (role, text) with role "user" | "assistant"; each backend converts it to its own format.
history: list[tuple[str, str]] = []  # single-user demo: one conversation


def turn(role: str, text: str) -> tuple[str, str]:
    return (role, text)


# Both are loaded at startup; the web UI picks one per conversation. A failed load only greys that choice out.
LLMS = {
    "gemini": {"label": "Gemini 3.5 Flash-Lite (cloud)", "model": GEMINI_MODEL, "ok": False},
    "gemma": {"label": "Gemma 4 12B, QAT 4-bit (local, llama.cpp)", "model": GEMMA_MODEL, "ok": False},
}
DEFAULT_LLM = "gemini"
active_llm = DEFAULT_LLM  # the one most recently used; recorded in the chat log

try:
    client = genai.Client(api_key=os.environ["GENAI_API_KEY"])
    LLM_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    LLMS["gemini"]["ok"] = True
except Exception as e:
    print(f"[llm] Gemini unavailable: {e!r} (set GENAI_API_KEY)", flush=True)

llama_proc = None


def start_llama_server():
    """Start llama.cpp's server with Gemma and wait until it is ready. Its log goes to llama_server.log."""
    global llama_proc
    from huggingface_hub import hf_hub_download
    if not Path(LLAMA_SERVER).exists():
        raise FileNotFoundError(f"{LLAMA_SERVER} not found (build llama.cpp, or set LLAMA_SERVER)")
    gguf = hf_hub_download(GEMMA_MODEL, GEMMA_FILE)
    cmd = [LLAMA_SERVER, "-m", gguf, "-ngl", "99", "-c", str(LLAMA_CTX), "-np", "1", "--host", "127.0.0.1",
           "--port", str(LLAMA_PORT), "--jinja", "-fa", "on", "--chat-template-kwargs", '{"enable_thinking":false}',
           "--swa-full"]  # Gemma's sliding-window layers can't reuse the cached conversation start without this (+2 GB, -0.3 s/turn)
    llama_proc = subprocess.Popen(cmd, stdout=open(HERE / "llama_server.log", "w"), stderr=subprocess.STDOUT)
    atexit.register(stop_llama_server)
    for _ in range(240):  # up to 2 minutes
        if llama_proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {llama_proc.returncode} (see llama_server.log)")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=1).status == 200:
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("llama-server did not become ready (see llama_server.log)")


def stop_llama_server():
    if llama_proc and llama_proc.poll() is None:
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_proc.kill()


try:
    start_llama_server()
    LLMS["gemma"]["ok"] = True
except Exception as e:
    print(f"[llm] Gemma unavailable: {e!r}", flush=True)
if not LLMS[DEFAULT_LLM]["ok"]:
    DEFAULT_LLM = active_llm = next((k for k, v in LLMS.items() if v["ok"]), DEFAULT_LLM)


def _gemini_stream(messages, cancelled):
    contents = [types.Content(role="model" if r == "assistant" else r, parts=[types.Part(text=t)]) for r, t in messages]
    for chunk in client.models.generate_content_stream(model=GEMINI_MODEL, contents=contents, config=LLM_CONFIG):
        if cancelled():
            return
        if chunk.text:
            yield chunk.text


def _gemma_stream(messages, cancelled):
    """Stream from the llama.cpp server. Dropping the connection makes the server stop generating at once."""
    chat = [{"role": "system", "content": SYSTEM_PROMPT}] + [{"role": r, "content": t} for r, t in messages[-GEMMA_HISTORY_MESSAGES:]]
    body = {"model": "gemma", "messages": chat, "stream": True, "max_tokens": GEMMA_MAX_NEW_TOKENS,
            "temperature": 1.0, "top_k": 64, "top_p": 0.95,   # Google's recommended sampling for Gemma 4
            "cache_prompt": True,                              # reuse the shared start of the conversation between turns
            "chat_template_kwargs": {"enable_thinking": False}}  # answer right away, no reasoning delay
    req = urllib.request.Request(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    try:
        for raw in resp:  # server-sent events: b"data: {...}\n"
            if cancelled():
                return
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            choices = json.loads(data).get("choices") or [{}]
            piece = choices[0].get("delta", {}).get("content")
            if piece:
                yield piece
    finally:
        resp.close()


def llm_stream(messages, cancelled=lambda: False, llm=None):
    """Yield the reply text piece by piece from the chosen LLM ("gemini" | "gemma")."""
    global active_llm
    llm = llm or DEFAULT_LLM
    if llm not in LLMS or not LLMS[llm]["ok"]:
        raise RuntimeError(f"LLM {llm!r} is not available")
    active_llm = llm
    yield from (_gemini_stream if llm == "gemini" else _gemma_stream)(messages, cancelled)


# Full duplex: a newer request (or /api/interrupt) bumps `turn_no`, which makes the running reply stop.
turn_no = 0
turn_done = threading.Event()  # set when the latest reply generator has finished
turn_done.set()
turn_lock = threading.Lock()


# ---------------- session logs ----------------
# logs/normal/<time>/        real conversations
# logs/debug/debug_<time>/   test-mode runs
#   chatlog.jsonl + chatlog.txt  (roles: user, backchannel, ai, event)
#   audio/human/NNNN.wav         what you said, per utterance (incl. backchannels)
#   audio/ai/NNNN.wav            each synthesized sentence
#   audio/human_full_NN.wav      continuous mic recording
#   audio/ai_full_NN.wav         AI voice on the same timeline: silence except where it was actually heard
# All disk writes go through one background thread, so the request path only does a queue.put().
LOG_ROOT = HERE / "logs"
log_q: queue.Queue = queue.Queue()
log_lock = threading.Lock()
session = {"dir": None, "kind": "normal", "n": {"human": 0, "ai": 0, "full": 0}}
mic_rec = {"rel": None, "path": None, "sr": 0, "samples": 0, "ai_rel": None, "ai_path": None}
# ^ the continuous human recording being appended to, and the matching time-aligned AI track


def finalize_recording_locked():
    """Pad the AI track with trailing silence so it ends when the human recording ends (call with log_lock held)."""
    if mic_rec["ai_path"] and mic_rec["sr"]:
        log_q.put(("ai_pad", mic_rec["ai_path"], mic_rec["samples"] / mic_rec["sr"]))


# Final lengthening factors per kind of phrase ending. The ORDER (clause-final particles lengthen more at deeper
# boundaries; ne longer than yo) follows Den 2015 and a 2025 J. Japanese Linguistics study; the VALUES are design
# choices tuned by ear (no paper gives a per-particle ratio). Editable from the Debug tab.
STRETCH_DEFAULTS = {"ne_final": 1.9, "yo_final": 1.6, "sa_final": 1.7, "link": 1.4, "ne_mid": 1.6, "yo_mid": 1.6, "sa_mid": 1.6}
STRETCH_LABELS = [  # (key, label, example): final = ends the sentence (。), mid = followed by more (、)
    ("ne_final", "ね at sentence end", "楽しかったね。"), ("yo_final", "よ at sentence end", "おいしかったよ。"),
    ("sa_final", "さ at sentence end", "そうだったさ。"), ("ne_mid", "ね mid-sentence", "昨日ね、"),
    ("yo_mid", "よ mid-sentence", "ほんとだよ、"), ("sa_mid", "さ mid-sentence", "行ってさ、"),
    ("link", "clause link (て / けど / が)", "寄ったんだけど、"),
]
STRETCH_JITTER = 0.1  # each stretched chunk gets its factor +- up to this much (variety within a conversation)
PARTICLE_KEY = {"ね": "ne", "よ": "yo", "さ": "sa"}
ENDING_RE = re.compile(r"(ね|よ|さ|けど|けれど|が|て)([。、！？!?]?)$")


def classify_ending(text: str) -> str | None:
    m = ENDING_RE.search(text.strip())
    if not m:
        return None
    part, punct = m.groups()
    if part in ("けど", "けれど", "が", "て"):
        return "link"
    return f"{PARTICLE_KEY[part]}_{'mid' if punct == '、' else 'final'}"


class TailStretcher:
    """Picks the final-lengthening factor of a chunk from its ending (table) plus a small random jitter drawn from a
    per-conversation generator, so factors vary within and across conversations."""
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.eligible, self.factors = 0, []

    def decide(self, text: str, table: dict | None, jitter: float) -> tuple[str, float] | None:
        key = classify_ending(text)
        if not key:
            return None
        self.eligible += 1
        base = float((table or {}).get(key, STRETCH_DEFAULTS[key]))
        if base <= 1.0:
            return None
        f = round(max(1.0, base + self.rng.uniform(-jitter, jitter)), 2)
        if f <= 1.0:
            return None
        self.factors.append(f)
        return key, f


tail_stretcher = TailStretcher()  # one per conversation: replaced in new_session()


def new_session(kind: str | None = None):
    """Start a fresh log folder. Normal chats: logs/normal/<time>; debug/test runs: logs/debug/debug_<time>."""
    global tail_stretcher
    tail_stretcher = TailStretcher()
    with log_lock:
        if kind:
            session["kind"] = kind
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = ts if session["kind"] == "normal" else f"debug_{ts}"
        session["dir"] = LOG_ROOT / session["kind"] / name  # created on first write
        session["n"] = {"human": 0, "ai": 0, "full": 0}
        finalize_recording_locked()
        mic_rec.update(rel=None, path=None, sr=0, samples=0, ai_rel=None, ai_path=None)  # next mic chunk starts a new recording


def log_audio(role: str, wav: bytes, hold: list | None = None) -> str:
    """Queue a WAV for saving; returns its path relative to the session folder.
    With `hold`, the write is parked in that list instead (speculative replies; see /api/spec/commit)."""
    with log_lock:
        session["n"][role] += 1
        rel = f"audio/{role}/{session['n'][role]:04d}.wav"
        item = ("file", session["dir"] / rel, wav)
        if hold is None:
            log_q.put(item)
        else:
            hold.append(item)
    return rel


def log_chat(role: str, text: str, audio: str | None = None, hold: list | None = None, **extra):
    with log_lock:
        entry = {"time": datetime.now().isoformat(timespec="milliseconds"), "mode": session["kind"], "llm": LLMS[active_llm]["model"],
                 "role": role, "text": text, "audio": audio, **extra}
        item = ("chat", session["dir"], entry)
        if hold is None:
            log_q.put(item)
        else:
            hold.append(item)


ai_track_sr: dict[Path, int] = {}  # AI track path -> sample rate (writer thread only)


def wav_header(sr: int, data_len: int) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", data_len))


def patch_wav_size(f, data_len: int):
    f.seek(4); f.write(struct.pack("<I", 36 + data_len))
    f.seek(40); f.write(struct.pack("<I", data_len))


def place_ai_clip(track: Path, clip: Path, start_s: float, cut_s: float | None):
    """Write `clip` into the silent AI track at start_s. cut_s = how long it actually played if it was interrupted."""
    data, sr = sf.read(clip, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    if cut_s is not None:  # interrupted: keep only what was heard, with a 10 ms fade-out to avoid a click
        data = data[: int(max(cut_s, 0) * sr)].copy()
        fade = min(len(data), int(0.01 * sr))
        if fade:
            data[-fade:] = (data[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    if track not in ai_track_sr:
        track.parent.mkdir(parents=True, exist_ok=True)
        track.write_bytes(wav_header(sr, 0))
        ai_track_sr[track] = sr
    elif ai_track_sr[track] != sr:  # different TTS model mid-recording: resample to the track's rate
        n = int(len(data) * ai_track_sr[track] / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.int16)
        sr = ai_track_sr[track]
    pos = int(round(start_s * sr))
    with open(track, "r+b") as f:
        f.seek(0, 2)
        size = f.tell() - 44
        f.seek(44 + pos * 2)  # past the end = the gap is filled with zeros (silence)
        f.write(data.tobytes())
        patch_wav_size(f, max(size, (pos + len(data)) * 2))


def pad_ai_track(track: Path, seconds: float):
    if track not in ai_track_sr:
        return
    want = int(seconds * ai_track_sr[track]) * 2
    with open(track, "r+b") as f:
        f.seek(0, 2)
        size = f.tell() - 44
        if want > size:
            f.seek(44 + want - 1); f.write(b"\0")
            patch_wav_size(f, want)


def log_writer():
    while (item := log_q.get()) is not None:
        try:
            kind, where, payload = item
            if kind == "file":
                where.parent.mkdir(parents=True, exist_ok=True)
                where.write_bytes(payload)
            elif kind == "mic_new":  # WAV header with empty data; sizes are patched after every chunk
                where.parent.mkdir(parents=True, exist_ok=True)
                where.write_bytes(wav_header(payload, 0))
            elif kind == "mic":  # append 16-bit mono PCM, keep the file a valid WAV at all times
                with open(where, "r+b") as f:
                    f.seek(0, 2)
                    f.write(payload)
                    patch_wav_size(f, f.tell() - 44)
            elif kind == "ai_place":  # put one played AI clip at its real playback time in the AI track
                place_ai_clip(where, *payload)
            elif kind == "ai_pad":
                pad_ai_track(where, payload)
            else:
                where.mkdir(parents=True, exist_ok=True)
                with open(where / "chatlog.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                with open(where / "chatlog.txt", "a", encoding="utf-8") as f:
                    f.write(f"[{payload['time'][11:23]}] {payload['role'].upper()}: {payload['text']}\n")
        except Exception as e:
            print(f"[log] write failed: {e!r}", flush=True)


log_thread = threading.Thread(target=log_writer, daemon=True)
log_thread.start()
new_session()


def warmup():
    """Run each model once so the first real turn doesn't pay CUDA / lazy-load cost. Failures are non-fatal."""
    def step(name, fn):
        t0 = time.perf_counter()
        try:
            fn()
            print(f"[warmup] {name} ready ({time.perf_counter() - t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[warmup] {name} skipped: {e!r}", flush=True)

    def asr_step():
        asr({"raw": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}, generate_kwargs=ASR_KWARGS)

    def tts_step():
        model_id = os.environ.get("TTS_MODEL") or list_tts_models()[0]["id"]
        synth(TTSRequest(text="こんにちは。今日はいい天気ですね。", model=model_id))

    def filler_step():  # cache the backchannel phrases for the default voice so they play instantly
        model_id = os.environ.get("TTS_MODEL") or list_tts_models()[0]["id"]
        for text in FILLERS:
            req = TTSRequest(text=text, model=model_id)
            _filler_cache[(req.model, req.style, req.style_weight, req.length, text)] = synth(req)

    def llm_step(llm):
        for _ in llm_stream([turn("user", "こんにちは")], llm=llm):
            pass

    step("ASR", asr_step)
    step("TTS", tts_step)
    step("fillers", filler_step)
    for name, info in LLMS.items():
        if info["ok"]:
            step(f"LLM {name}", lambda name=name: llm_step(name))


@asynccontextmanager
async def lifespan(_app):
    await asyncio.to_thread(warmup)  # the server starts accepting requests only after this
    yield
    with log_lock:
        finalize_recording_locked()
    log_q.put(None)  # flush pending log writes
    stop_llama_server()
    log_thread.join(timeout=10)


app = FastAPI(lifespan=lifespan)


def decode_audio(data: bytes) -> np.ndarray:
    """Browser recording (webm/ogg/mp4) -> 16 kHz mono float32 via ffmpeg."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=data, capture_output=True,
    )
    if p.returncode != 0:
        raise HTTPException(400, "could not decode audio")
    return np.frombuffer(p.stdout, dtype=np.float32)


class ChatRequest(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    model: str
    style: str = "Neutral"
    style_weight: float = 5.0
    length: float = 1.0  # >1.0 slower
    stretch_table: dict[str, float] | None = None  # final-lengthening factor per ending kind (None = STRETCH_DEFAULTS)
    stretch_jitter: float = STRETCH_JITTER


@app.get("/api/llms")
def api_llms():
    return {"default": DEFAULT_LLM, "llms": [{"id": k, "label": v["label"], "ok": v["ok"]} for k, v in LLMS.items()]}


@app.get("/api/models")
def api_models():
    return list_tts_models()


def log_time(name: str, t0: float, extra: str = ""):
    print(f"[time] {name}: {time.perf_counter() - t0:.2f}s {extra}".rstrip(), flush=True)


@app.post("/api/mic_chunk")
async def api_mic_chunk(request: Request, sr: int, start: int = 0):
    """Continuous mic recording: browser posts raw 16-bit mono PCM about once a second; appended to one WAV."""
    pcm = await request.body()
    with log_lock:
        if start or mic_rec["path"] is None:  # new recording (mic switched on, or chat reset)
            finalize_recording_locked()
            session["n"]["full"] += 1
            n = session["n"]["full"]
            rel, ai_rel = f"audio/human_full_{n:02d}.wav", f"audio/ai_full_{n:02d}.wav"
            mic_rec.update(rel=rel, path=session["dir"] / rel, sr=sr, samples=0,
                           ai_rel=ai_rel, ai_path=session["dir"] / ai_rel)
            log_q.put(("mic_new", mic_rec["path"], sr))
        mic_rec["samples"] += len(pcm) // 2
        log_q.put(("mic", mic_rec["path"], pcm))
    return {"ok": True}


@app.post("/api/mic_end")
def api_mic_end():
    """Mic switched off: close the AI track at the same length as the human recording."""
    with log_lock:
        finalize_recording_locked()
    return {"ok": True}


class PlayEvent(BaseModel):
    id: str  # the AI clip, e.g. "audio/ai/0007.wav"
    start_s: float  # when it really started playing, on the mic recording's clock
    end_s: float  # when it stopped (finished, or cut off by an interruption)
    completed: bool


@app.post("/api/play_event")
def api_play_event(ev: PlayEvent):
    """Browser reports when an AI clip was actually heard; place it there in the time-aligned AI track."""
    with log_lock:
        if not mic_rec["ai_path"]:
            return {"ok": False}  # mic recording not running, so there is no timeline to align to
        clip = (session["dir"] / ev.id).resolve()
        if (session["dir"] / "audio" / "ai").resolve() not in clip.parents:
            raise HTTPException(400, "bad clip id")
        log_q.put(("ai_place", mic_rec["ai_path"], (clip, ev.start_s, None if ev.completed else ev.end_s - ev.start_s)))
        ai_rel = mic_rec["ai_rel"]
    log_chat("event", f"ai played {ev.start_s:.2f}-{ev.end_s:.2f}s ({'completed' if ev.completed else 'interrupted'})",
             clip=ev.id, full_audio=ai_rel, full_offset_s=round(ev.start_s, 3), end_s=round(ev.end_s, 3), completed=ev.completed)
    return {"ok": True}


# A clip whose transcript is only an ellipsis / punctuation, or only aizuchi words, is an aizuchi, not a turn: the page must not answer it.
BACKCHANNEL_WORDS = re.compile(r"(?:う+ん+|ん+|ふ+ん+|は+い+|え+え*|へ+え+|あ+|お+|そう|なるほど)+")


def is_backchannel_text(text: str, seconds: float) -> bool:
    t = re.sub(r"[\s…。、，．,.！？!?・〜~ー]", "", text)  # (ー removed so うーん = うん)
    if t and re.search(r"[?？]", text):  # "え？" is a request to repeat, not an aizuchi
        return False
    return t == "" or (seconds <= 4 and bool(BACKCHANNEL_WORDS.fullmatch(t)))


@app.post("/api/asr")
def api_asr(audio: UploadFile = File(...), kind: str = Form("user"), offset: float | None = Form(None),
            log: int = Form(1), known_text: str | None = Form(None)):
    # kind: "user" | "backchannel"; offset: where this utterance starts in the continuous mic recording (seconds)
    # log=0: speculative transcription, don't record it; known_text: already transcribed speculatively, skip the model
    t0 = time.perf_counter()
    raw = audio.file.read()
    samples = decode_audio(raw)
    if len(samples) < 1600:  # < 0.1 s
        return {"text": ""}
    asr_ms = None
    if known_text is not None:
        text = known_text.strip()
    else:
        t_asr = time.perf_counter()
        result = asr({"raw": samples, "sampling_rate": 16000}, generate_kwargs=ASR_KWARGS)
        text = result["text"].strip()
        asr_ms = round((time.perf_counter() - t_asr) * 1000)
    log_time("ASR", t0, f"(audio {len(samples) / 16000:.1f}s{', reused speculative text' if known_text is not None else ''}"
                        f"{'' if log else ', speculative'}) -> {text!r}")
    kind = kind if kind in ("user", "backchannel") else "user"
    reclassified = kind == "user" and bool(text) and is_backchannel_text(text, len(samples) / 16000)
    if reclassified:
        kind = "backchannel"
    if text and log:
        with log_lock:
            full = mic_rec["rel"]
        log_chat(kind, text, log_audio("human", raw), **({"reclassified": True} if reclassified else {}),
                 full_audio=full if offset is not None else None,
                 full_offset_s=round(offset, 3) if offset is not None and full else None,
                 duration_s=round(len(samples) / 16000, 3), asr_ms=asr_ms, asr_reused=known_text is not None)
    elif text and not log:  # speculative transcription: not a chat line, but its cost is worth keeping
        log_chat("event", "speculative ASR", asr_ms=asr_ms, audio_s=round(len(samples) / 16000, 2))
    return {"text": text, "backchannel": kind == "backchannel"}


@app.get("/api/debug/log")
def api_debug_log():
    """Current session's chat log (as written to disk) for the Debug tab."""
    with log_lock:
        d = session["dir"]
    entries = []
    f = d / "chatlog.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                pass  # half-written last line
    return {"session": f"{d.parent.name}/{d.name}", "pending_writes": log_q.qsize(), "entries": entries}


# ---------------- ASR comparison (Debug tab) ----------------
# Candidate models are loaded on first use and kept; "anime" is the one the chat already runs (ASR_MODEL).
ASR_CANDIDATES = {
    "kotoba": ("kotoba-tech/kotoba-whisper-v2.0", {}),
    "anime": ("litagin/anime-whisper", {"no_repeat_ngram_size": 0, "repetition_penalty": 1.0}),  # per its model card: no initial prompt
    "turbo": ("openai/whisper-large-v3-turbo", {}),
}
_asr_models: dict[str, object] = {"anime": asr}
_asr_lock = threading.Lock()


def get_asr(key: str):
    """A candidate key, or any 'org/name' Hugging Face id (Whisper-style, loaded with the plain transformers pipeline)."""
    with _asr_lock:
        if key not in _asr_models:
            if key in ASR_CANDIDATES:
                model_id = ASR_CANDIDATES[key][0]
            elif re.fullmatch(r"[\w.-]+/[\w.-]+", key):
                model_id = key
            else:
                raise HTTPException(400, f"unknown ASR model {key!r}")
            print(f"[asr-compare] loading {model_id} ...", flush=True)
            pipe = pipeline("automatic-speech-recognition", model=model_id, device=DEVICE, chunk_length_s=15,
                            dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
            try:  # one throwaway run so the first timed transcription is not slowed by GPU warm-up
                pipe({"raw": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}, generate_kwargs={"language": "ja", "task": "transcribe"})
            except Exception:
                pass
            _asr_models[key] = pipe
        return _asr_models[key]


def run_asr_models(samples: np.ndarray, loaded: dict) -> dict:
    """Transcribe one clip with every loaded model; ms is the transcription time only (the model is already loaded)."""
    results = {}
    for m, pipe in loaded.items():
        kw = {"language": "ja", "task": "transcribe", **(ASR_CANDIDATES.get(m, ("", {}))[1])}
        t = time.perf_counter()
        try:
            text = pipe({"raw": samples, "sampling_rate": 16000}, generate_kwargs=kw)["text"].strip()
        except Exception as e:
            text = f"[error: {e!r}]"
        results[m] = {"text": text, "ms": round((time.perf_counter() - t) * 1000)}
    return results


@app.get("/api/asr_models")
def api_asr_models():
    return [{"key": k, "id": v[0], "loaded": k in _asr_models} for k, v in ASR_CANDIDATES.items()]


@app.get("/api/asr_clips")
def api_asr_clips(role: str = "backchannel", max_s: float = 3.0, limit: int = 200):
    """Saved human clips from every session's log (newest first), with the text the chat's ASR gave them."""
    out = []
    for f in sorted(LOG_ROOT.glob("**/chatlog.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line) if line.strip() else {}
                if r.get("role") != role or not r.get("audio") or (r.get("duration_s") or 0) > max_s:
                    continue
                rel = (f.parent / r["audio"]).relative_to(LOG_ROOT).as_posix()
                out.append({"rel": rel, "text": r.get("text", ""), "seconds": r.get("duration_s"),
                            "session": f.parent.name})
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out[:limit]


@app.get("/api/asr_clip_audio/{rel:path}")
def api_asr_clip_audio(rel: str):
    base = LOG_ROOT.resolve()
    path = (base / rel).resolve()
    if base not in path.parents or not path.is_file() or path.suffix != ".wav":
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/asr_compare_upload")
def api_asr_compare_upload(audio: UploadFile = File(...), models: str = Form(...)):
    """Live test: a fresh mic recording (any browser format) run through the chosen models."""
    samples = decode_audio(audio.file.read())
    keys = [m for m in models.split(",") if m.strip()]
    loaded = {m: get_asr(m.strip()) for m in keys}
    return {"seconds": len(samples) / 16000, "results": run_asr_models(samples, loaded)}


class AsrCompareRequest(BaseModel):
    clips: list[str]  # paths relative to LOG_ROOT, as listed by /api/asr_clips
    models: list[str]


@app.post("/api/asr_compare")
def api_asr_compare(req: AsrCompareRequest):
    base = LOG_ROOT.resolve()
    loaded = {}
    for m in req.models:
        t = time.perf_counter()
        loaded[m] = get_asr(m)
        if time.perf_counter() - t > 1:
            print(f"[asr-compare] {m} ready in {time.perf_counter() - t:.1f}s", flush=True)
    out = []
    for rel in req.clips:
        path = (base / rel).resolve()
        if base not in path.parents or not path.is_file():
            out.append({"rel": rel, "error": "not found"})
            continue
        samples = decode_audio(path.read_bytes())
        row = {"rel": rel, "seconds": len(samples) / 16000, "results": run_asr_models(samples, loaded)}
        out.append(row)
    return {"results": out}


@app.get("/api/debug/audio/{rel:path}")
def api_debug_audio(rel: str):
    with log_lock:
        base = (session["dir"] / "audio").resolve()
    path = (base / rel).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/session")
def api_session(kind: str):
    """Switch between the normal and debug (test mode) log folders; starts a fresh session either way."""
    if kind not in ("normal", "debug"):
        raise HTTPException(400, "kind must be normal or debug")
    new_session(kind)
    return {"session": f"{session['dir'].parent.name}/{session['dir'].name}"}


@app.post("/api/typed")
def api_typed(req: ChatRequest):
    """Typed messages don't pass through ASR; just record them."""
    log_chat("user", req.text.strip(), typed=True)
    return {"ok": True}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    t0 = time.perf_counter()
    user = turn("user", req.text.strip())
    reply = "".join(llm_stream(history + [user])).strip()
    history.extend([user, turn("assistant", reply)])
    log_time("LLM", t0, f"-> {reply!r}")
    return {"reply": reply}


def trim_silence(wav: np.ndarray, sr: int, thr: float = 0.01, lead_ms: int = 30, tail_ms: int = 100) -> np.ndarray:
    """Cut the synthesizer's built-in padding (about 0.4 s of faint noise before and after every sentence).
    Measured on real clips: only the padding is removed (at most 0.01% of a clip's energy), so the gap between
    sentences is then set by the page's "Pause between sentences" instead of ~0.9 s of baked-in silence."""
    w = max(1, int(0.01 * sr))
    mag = np.abs(wav.astype(np.float32))
    if np.issubdtype(wav.dtype, np.integer):  # the voice model returns 16-bit integers: measure on a -1..1 scale
        mag /= np.iinfo(wav.dtype).max
    level = np.convolve(mag, np.ones(w) / w, mode="same")  # 10 ms average level
    loud = np.where(level > thr)[0]
    if not len(loud):
        return wav
    return wav[max(0, loud[0] - lead_ms * sr // 1000): min(len(wav), loud[-1] + 1 + tail_ms * sr // 1000)]


# Final lengthening on the last particle of a chunk: the chunk is synthesized in ONE pass (so it stays continuous),
# then the last TAIL_MS of the audio is time-stretched by TAIL_STRETCH (pitch kept) and cross-faded back in.
TAIL_PARTICLE = re.compile(r"(ね|さ|よ|けど|けれど|が|て)[。、！？!?]?$")
TAIL_MS, TAIL_STRETCH, TAIL_FADE_MS = 200, 2.0, 20


def lengthen_tail(wav: np.ndarray, sr: int, tail_ms: int = TAIL_MS, stretch: float = TAIL_STRETCH) -> np.ndarray:
    n, f = int(tail_ms * sr / 1000), int(TAIL_FADE_MS * sr / 1000)
    if len(wav) < 3 * n:
        return wav
    x = wav.astype(np.float32)
    tail = librosa.effects.time_stretch(x[-n - f:], rate=1 / stretch)  # includes the cross-fade overlap
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    head = x[:-n - f]
    mid = x[-n - f:-n] * (1 - ramp) + tail[:f] * ramp  # blend original -> stretched
    out = np.concatenate([head, mid, tail[f:]])
    return out.astype(wav.dtype) if np.issubdtype(wav.dtype, np.integer) else out


def synth_raw(req: TTSRequest) -> tuple[np.ndarray, int]:
    """Text -> (trimmed samples, sample rate), before any final lengthening."""
    model = get_tts(req.model)
    style = req.style if req.style in model.style2id else next(iter(model.style2id))
    # SBV2's g2p crashes on "!" before more text ("Input must be katakana only: ！"), so use "。" instead.
    text = req.text.replace("！", "。").replace("!", "。")
    sr, wav = model.infer(
        text=text, language=Languages.JP, style=style, style_weight=req.style_weight, length=req.length,
    )
    return trim_silence(wav, sr), sr


def to_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def synth(req: TTSRequest, stretch: float | None = None) -> bytes:
    """Text -> WAV bytes (silence at both ends trimmed; the end lengthened by factor `stretch` when given)."""
    wav, sr = synth_raw(req)
    if stretch and stretch > 1.0:
        wav = lengthen_tail(wav, sr, stretch=stretch)
    return to_wav_bytes(wav, sr)


@app.get("/api/stretch_defaults")
def api_stretch_defaults():
    return {"table": STRETCH_DEFAULTS, "jitter": STRETCH_JITTER,
            "labels": [{"key": k, "label": l, "example": e} for k, l, e in STRETCH_LABELS]}


SENTENCE_ONLY_END = re.compile(r"[^。！？!?\n]*[。！？!?\n]+")  # the old rule: sentence-level chunks only


def split_chunks(text: str, pattern: re.Pattern) -> list[str]:
    """Same loop as api_chat_stream: peel chunks off the front; whatever is left at the end is the last chunk."""
    out, buf = [], text
    while (m := pattern.match(buf)):
        out.append(m.group().strip())
        buf = buf[m.end():]
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if SPEAKABLE.search(c)]


class ChunkTestRequest(TTSRequest):
    text: str = ""  # unused (inherited): the replies come from `texts`
    texts: list[str]


@app.post("/api/chunk_test")
def api_chunk_test(req: ChunkTestRequest):
    """Debug: split each reply into sentence-level chunks (old rule) and phrase-level chunks (new rule), synthesize
    every chunk exactly as live chat does (final lengthening included), so the two can be played with the same pause."""
    def build(chunks, stretcher=None):
        clips = []
        for c in chunks:
            d = stretcher.decide(c, req.stretch_table, req.stretch_jitter) if stretcher else None
            wav = synth(req.model_copy(update={"text": c}), stretch=d[1] if d else None)
            with sf.SoundFile(io.BytesIO(wav)) as f:
                dur = len(f) / f.samplerate
            clips.append({"text": c, "seconds": dur, "audio": base64.b64encode(wav).decode(),
                          "factor": d[1] if d else None, "kind": d[0] if d else None})
        return clips
    out = []
    for text in req.texts:  # one fresh stretcher per reply: it mimics the start of one conversation
        text = text.strip()
        if text:
            phrase = split_chunks(text, SENTENCE_END)
            st = TailStretcher()
            out.append({"text": text,
                        "sentence": build(split_chunks(text, SENTENCE_ONLY_END)),
                        "phrase": build(phrase),
                        "phrase_q": build(phrase, st), "eligible": st.eligible, "factors": st.factors})
    return {"results": out}


class LengthenTestRequest(TTSRequest):
    text: str = ""  # unused (inherited, but the texts come from `texts`)
    texts: list[str]
    tail_ms: int = TAIL_MS
    stretch: float = TAIL_STRETCH


@app.post("/api/lengthen_test")
def api_lengthen_test(req: LengthenTestRequest):
    """Debug: for each text, ONE synthesis, returned both as-is and with final lengthening, so the two differ only by the stretch."""
    out = []
    for text in req.texts:
        text = text.strip()
        if not text:
            continue
        wav, sr = synth_raw(req.model_copy(update={"text": text}))
        applies = bool(TAIL_PARTICLE.search(text))
        long = lengthen_tail(wav, sr, req.tail_ms, req.stretch) if applies else wav
        out.append({"text": text, "applies": applies, "sr": sr,
                    "plain_s": len(wav) / sr, "long_s": len(long) / sr,
                    "plain": base64.b64encode(to_wav_bytes(wav, sr)).decode(),
                    "long": base64.b64encode(to_wav_bytes(long, sr)).decode(),
                })
    return {"results": out}


# Fixed "I'm listening" phrases the page plays when you pause briefly mid-turn (no LLM involved).
FILLERS = ["うん。", "うんうん。", "うん、うん。"]
_filler_cache: dict[tuple, bytes] = {}


@app.post("/api/filler")
def api_filler(req: TTSRequest):
    """Synthesize (and cache) one of FILLERS; logged like any AI clip so it also lands in the AI full track."""
    if req.text not in FILLERS:
        raise HTTPException(400, "not a known filler")
    key = (req.model, req.style, req.style_weight, req.length, req.text)
    if key not in _filler_cache:
        _filler_cache[key] = synth(req)
    audio = _filler_cache[key]
    clip_id = log_audio("ai", audio)
    log_chat("ai", req.text, clip_id, backchannel=True)
    return {"text": req.text, "audio": base64.b64encode(audio).decode(), "id": clip_id}


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    t0 = time.perf_counter()
    audio = synth(req)
    log_time("TTS", t0)
    return Response(audio, media_type="audio/wav")


# Phrase chunking: a chunk ends at a sentence end, or at a comma that follows a final particle / clause-linking ending
# (ne, sa, yo, -te, -kedo, ga). Each chunk is its own audio clip, so the page's fixed pause after it is the aizuchi slot.
SENTENCE_END = re.compile(r"[^。！？!?\n]*?(?:(?:ね|さ|よ|て|けど|けれど|が)、|[。！？!?\n]+)")


# ---- LLM latency benchmark (Debug tab) ----
# Same prompts for every LLM, run one after another. "first sentence" = when the first complete sentence is available,
# which is what actually gates TTS. Run it while idle: it shares the GPU with everything else.
BENCH_PROMPTS = [
    [("user", "こんにちは")],
    [("user", "一度行ってみたい旅行先ってありますか？"), ("assistant", "私は海が見えるカフェのある地中海沿岸に行ってみたいな。"),
     ("user", "僕は北海道に行ってみたいんですよね。")],
    [("user", "こんにちは"), ("assistant", "やっほー、今日は何の話をしようか？"), ("user", "旅行の話がしたいです。"),
     ("assistant", "いいね！行ってみたい国とかある？"), ("user", "ヨーロッパかな。"),
     ("assistant", "ヨーロッパいいよね。私は街並みをのんびり歩きたいなあ。"),
     ("user", "なんかどこのふりのかは忘れたんですけどコピコっていうキャンディーそのキャンディーがなんかコーヒー味のキャンディで"
              "作業する時によく食べます。まあ、苦すぎなくて、少し甘いの方がいいんです。")],
]
bench_lock = threading.Lock()


@app.post("/api/debug/llm_bench")
def api_llm_bench(reps: int = 3):
    reps = max(1, min(reps, 10))
    if not bench_lock.acquire(blocking=False):
        raise HTTPException(409, "a benchmark is already running")
    try:
        out = []
        for name, info in LLMS.items():
            entry = {"id": name, "label": info["label"], "ok": info["ok"], "runs": [], "errors": []}
            out.append(entry)
            if not info["ok"]:
                continue
            try:
                list(llm_stream([("user", "こんにちは")], llm=name))  # warm-up, not counted
            except Exception as e:
                entry["errors"].append(f"warm-up: {e!r}")
            for _ in range(reps):
                for pi, msgs in enumerate(BENCH_PROMPTS):
                    nonce = len(entry["runs"]) + 1  # new text at the end each run, like a real new message (prefix may be cached)
                    msgs = msgs[:-1] + [(msgs[-1][0], msgs[-1][1] + "　" * nonce)]
                    t0, first, sent, buf = time.perf_counter(), None, None, ""
                    try:
                        for piece in llm_stream(msgs, llm=name):
                            now = time.perf_counter() - t0
                            first = now if first is None else first
                            buf += piece
                            if sent is None and SENTENCE_END.match(buf):
                                sent = now
                    except Exception as e:
                        entry["errors"].append(f"prompt {pi}: {e!r}")
                        continue
                    total = time.perf_counter() - t0
                    entry["runs"].append({"prompt": pi, "first": first, "sentence": total if sent is None else sent,
                                          "total": total, "chars": len(buf)})

            def med(key, rows):
                v = sorted(r[key] for r in rows if r[key] is not None)
                return v[len(v) // 2] if v else None
            entry["median"] = {k: med(k, entry["runs"]) for k in ("first", "sentence", "total", "chars")}
            entry["by_prompt"] = [{k: med(k, [r for r in entry["runs"] if r["prompt"] == pi]) for k in ("first", "sentence", "total")}
                                  for pi in range(len(BENCH_PROMPTS))]
            entry["worst_sentence"] = max((r["sentence"] for r in entry["runs"]), default=None)
        return {"reps": reps, "prompts": ["short greeting", "3-turn chat", "long 25 s utterance (6-turn history)"], "results": out}
    finally:
        bench_lock.release()
SPEAKABLE = re.compile(r"[^\W_]")  # at least one letter/kana/kanji/digit


# Sent to the LLM instead of a user message when the user has been silent for a while and the AI should break the silence.
CONTINUE_PROMPT = "（相手は黙っています。気まずくならないように、今の話の続きや、関連する思い出を、自分から話してください。）"


class StreamRequest(TTSRequest):
    cont: bool = False  # silence-breaking turn: `text` is ignored, CONTINUE_PROMPT is used and no user line is logged
    speculative: bool = False  # reply generated while you pause: nothing is logged/kept unless /api/spec/commit follows
    spec_id: str = ""
    llm: str = ""  # "gemini" | "gemma" (empty = default)
    test: bool = False  # debug: speak random canned sentences instead of calling the LLM
    # `text` is the user's message; the rest are TTS options


# Spoken when a reply fails completely (LLM error / empty answer), instead of leaving you in silence.
FALLBACK_REPLY = "ごめん、うまく聞き取れなかった。もう一回言ってくれる？"
LLM_ATTEMPTS = 3


# Debug test mode: long enough (several sentences, ~15 s) to leave room for backchannels while it talks.
TEST_REPLIES = [
    "今日はテスト用の文章を読み上げています。この間に、うんとか、へえとか、短い相づちを入れてみてください。"
    "相づちがログに残っているか、デバッグタブで確認できます。ゆっくり話すので、何度か試してみてね。",
    "昨日はね、近所の公園まで散歩に行ったんだ。桜がちょうど満開で、とてもきれいだったよ。"
    "それから、帰りに小さなパン屋さんでクロワッサンを買ったの。焼きたてで、すごくいい匂いがしたんだよ。",
    "これはランダムに選ばれたテスト文です。言語モデルは使っていません。"
    "あなたの声はそのまま録音されて、人間用のフォルダに保存されます。私の声は、別のフォルダに保存されます。",
    "最近、料理にはまっていて、毎日いろいろな味付けを試しているの。昨日は、トマトとバジルのスープを作ってみたよ。"
    "思ったより簡単で、お店の味みたいにできたんだ。今度は、カレーにも挑戦してみようかな。",
]


# Speculative replies: id -> {"committed": bool, "hold": [parked log writes]}
specs: dict[str, dict] = {}
spec_lock = threading.Lock()


@app.post("/api/spec/commit")
def api_spec_commit(id: str):
    """Your turn really ended: keep the speculative reply (write its parked log entries, log the rest as it comes)."""
    with spec_lock:
        st = specs.get(id)
        if not st:
            return {"ok": False}
        st["committed"] = True
        for item in st["hold"]:
            log_q.put(item)
        st["hold"].clear()
        if st.get("finished"):  # it was already fully generated: finish what the generator would have done
            specs.pop(id, None)
            if st["spoken"]:
                history.extend([st["user"], turn("assistant", "".join(st["spoken"]))])
            log_chat("event", "speculative reply committed", turn=st["turn"])
    return {"ok": True}


@app.post("/api/spec/cancel")
def api_spec_cancel(id: str):
    """You kept talking: drop the speculative reply."""
    with spec_lock:
        st = specs.pop(id, None)
    if st and st.get("finished"):  # (a still-running generator logs this itself when it notices the interrupt)
        print(f"[speculative] discarded after {len(st['spoken'])} sentence(s)", flush=True)
        log_chat("event", "speculative reply discarded (you kept talking)", sentences_ready=len(st["spoken"]), turn=st["turn"])
    return {"ok": True}


class LatencyEvent(BaseModel):
    latency_ms: int
    silence_wait_s: float
    speech_s: float
    early_reply: bool


@app.post("/api/latency")
def api_latency(ev: LatencyEvent):
    """The page measured how long after the turn was judged over the first sound of the reply started."""
    log_chat("event", "reply latency", latency_ms=ev.latency_ms, silence_wait_s=ev.silence_wait_s,
             speech_s=ev.speech_s, early_reply=ev.early_reply,
             from_last_word_s=round(ev.latency_ms / 1000 + ev.silence_wait_s, 2))
    return {"ok": True}


@app.post("/api/interrupt")
def api_interrupt():
    """Barge-in: tell the running reply to stop."""
    global turn_no
    with turn_lock:
        turn_no += 1
    return {"ok": True}


@app.post("/api/chat_stream")
def api_chat_stream(req: StreamRequest):
    """LLM streams text; each finished sentence is sent to TTS right away.
    Response is NDJSON: {"text": sentence, "audio": base64 wav} per line.
    Stops early if a newer request or /api/interrupt arrives; only the sentences already sent
    are kept in the history (and nothing is kept if none were sent)."""
    if not req.text.strip() and not req.test and not req.cont:
        raise HTTPException(400, "Empty text")
    spec = None
    if req.speculative:
        spec = specs[req.spec_id] = {"committed": False, "hold": []}

    def events():
        global turn_no, turn_done
        with turn_lock:
            turn_no += 1
            my_turn, prev_done, turn_done = turn_no, turn_done, threading.Event()
            done = turn_done
        prev_done.wait(timeout=5)  # let the cancelled reply commit its history first

        cancelled = lambda: my_turn != turn_no
        t0 = time.perf_counter()
        user = turn("user", CONTINUE_PROMPT if req.cont else req.text.strip())
        if req.cont:
            log_chat("event", "silence: AI continues the story", turn=my_turn)
        spoken, buf, first = [], "", True
        timing = {}  # seconds since this stream started: first LLM text, first spoken sentence, end

        def problem(msg, **extra):  # failures are recorded in the chat log (not only the console) so they can be found later
            print(f"[problem] {msg}", flush=True)
            log_chat("event", msg, turn=my_turn, **extra)

        def speak(sentence, fallback=False):
            sentence = sentence.strip()
            if not SPEAKABLE.search(sentence):
                return None
            t1 = time.perf_counter()
            try:
                d = None if fallback else tail_stretcher.decide(sentence, req.stretch_table, req.stretch_jitter)
                audio = synth(req.model_copy(update={"text": sentence}), stretch=d[1] if d else None)
            except Exception as e:  # one unspeakable sentence must not kill the whole reply
                traceback.print_exc()
                problem(f"TTS failed, sentence skipped: {e!r}", sentence=sentence)
                return None
            log_time("TTS", t1, f"-> {sentence!r} (since start {time.perf_counter() - t0:.2f}s)")
            if "first_clip_s" not in timing and not fallback:
                timing["first_clip_s"] = round(time.perf_counter() - t0, 2)
                timing["first_tts_s"] = round(time.perf_counter() - t1, 2)
            if not fallback:
                spoken.append(sentence)
            with spec_lock:  # while speculative and not committed, logging is parked; commit flushes it
                hold = spec["hold"] if spec and not spec["committed"] else None
                clip_id = log_audio("ai", audio, hold=hold)
                log_chat("ai", sentence, clip_id, hold=hold, turn=my_turn, stretch_kind=d[0] if d else None, stretch_factor=d[1] if d else None, **({"continued": True} if req.cont else {}), **({"fallback": True} if fallback else {}))
            return json.dumps({"text": sentence, "audio": base64.b64encode(audio).decode(), "id": clip_id}) + "\n"

        def llm_chunks():
            """LLM text chunks. An error or empty answer before any text arrived is retried (it has been seen to
            happen); after text has started, errors just end the reply."""
            if req.test:
                yield SimpleNamespace(text=random.choice(TEST_REPLIES))
                return
            for attempt in range(1, LLM_ATTEMPTS + 1):
                got = False
                try:
                    stream = llm_stream(history + [user], cancelled, req.llm or None)
                    try:
                        for piece in stream:
                            if cancelled():
                                return
                            got = True
                            yield SimpleNamespace(text=piece)
                    finally:
                        stream.close()  # stops the model's generation thread right away (matters for the local model)
                    if got:
                        return
                    reason = "empty answer"
                except Exception as e:
                    if got:
                        traceback.print_exc()
                        problem(f"LLM failed mid-reply: {e!r}")
                        return
                    traceback.print_exc()
                    reason = f"error {e!r}"
                problem(f"LLM {reason} (attempt {attempt}/{LLM_ATTEMPTS})", attempt=attempt)
                if attempt < LLM_ATTEMPTS:
                    time.sleep(0.5 * attempt)
                    if cancelled():
                        return

        try:
            for chunk in llm_chunks():
                if cancelled():
                    return
                if first:
                    log_time("LLM first chunk", t0)
                    timing["llm_first_s"] = round(time.perf_counter() - t0, 2)
                    first = False
                buf += chunk.text
                while (m := SENTENCE_END.match(buf)):
                    buf = buf[m.end():]
                    if (line := speak(m.group())) and not cancelled():
                        yield line
                    if cancelled():
                        return
            if (line := speak(buf)) and not cancelled():
                yield line
            if not spoken and not cancelled():  # nothing at all could be said: don't leave you in silence
                problem("no reply produced, speaking the fallback")
                if (line := speak(FALLBACK_REPLY, fallback=True)) and not cancelled():
                    yield line
            log_time("LLM+TTS total", t0)
            timing["total_s"] = round(time.perf_counter() - t0, 2)
        finally:
            with spec_lock:
                # A speculative reply that finished before your turn ended just waits: commit/cancel decides its fate.
                parked = bool(spec) and not spec["committed"] and not cancelled()
                if parked:
                    spec.update(finished=True, user=user, spoken=spoken, turn=my_turn)
                elif spec:
                    specs.pop(req.spec_id, None)
                discarded = bool(spec) and not spec["committed"] and not parked
            if parked:
                pass
            elif discarded:  # you kept talking: this reply never happened (nothing logged, history untouched)
                print(f"[speculative] discarded after {len(spoken)} sentence(s)", flush=True)
                log_chat("event", "speculative reply discarded (you kept talking)", sentences_ready=len(spoken), turn=my_turn)
            else:
                if spec:
                    log_chat("event", "speculative reply committed", turn=my_turn)
                if cancelled():
                    print(f"[interrupted] after {len(spoken)} sentence(s)", flush=True)
                    log_chat("event", "interrupted", sentences_sent=len(spoken), turn=my_turn)
                if spoken and not req.test:
                    history.extend([user, turn("assistant", "".join(spoken))])
            if timing and not cancelled():
                log_chat("event", "turn timing", turn=my_turn, speculative=req.speculative, cont=req.cont,
                         user_chars=len(req.text), sentences=len(spoken), **timing)
            done.set()

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/reset")
def reset():
    global turn_no
    with turn_lock:
        turn_no += 1
    history.clear()
    new_session()  # next messages go to a fresh log folder
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
