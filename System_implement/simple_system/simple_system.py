"""
Simple cascaded voice chat: mic -> Kotoba-Whisper (ASR) -> Gemini 3.5 Flash Lite (LLM) -> Style-Bert-VITS2 (TTS).

Run:  GENAI_API_KEY=... python simple_system.py   (then open http://localhost:8000)
Uses the same dependencies as ../Voice_Chat_App/requirements.txt.
"""

import asyncio
import base64
import io
import json
import os
import queue
import random
import struct
import subprocess
import sys
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
ASR_MODEL = "kotoba-tech/kotoba-whisper-v2.0"
LLM_MODEL = "gemini-3.5-flash-lite"
BERT_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"
HERE = Path(__file__).resolve().parent
MODEL_ASSETS = SBV2_DIR / "model_assets"

SYSTEM_PROMPT = """\
あなたは「あおい」、25歳の女性です。
相手は24歳の男性です。
ユーザーと一度行ってみたい旅行先の話しをします。
返事は1〜2文の短い話し言葉で答えてください。
質問以外、自分のことも話してください。
時々今の話題に近い話題に切り替えてもいいです。
相手が話し方言葉を繰り返さないようにしてください
"""

# ---------------- load models once at startup ----------------
asr = pipeline(
    "automatic-speech-recognition", model=ASR_MODEL, device=DEVICE, chunk_length_s=15,
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


client = genai.Client(api_key=os.environ["GENAI_API_KEY"])


LLM_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
history: list[types.Content] = []  # single-user demo: one conversation


def turn(role: str, text: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=text)])


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


def new_session(kind: str | None = None):
    """Start a fresh log folder. Normal chats: logs/normal/<time>; debug/test runs: logs/debug/debug_<time>."""
    with log_lock:
        if kind:
            session["kind"] = kind
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = ts if session["kind"] == "normal" else f"debug_{ts}"
        session["dir"] = LOG_ROOT / session["kind"] / name  # created on first write
        session["n"] = {"human": 0, "ai": 0, "full": 0}
        finalize_recording_locked()
        mic_rec.update(rel=None, path=None, sr=0, samples=0, ai_rel=None, ai_path=None)  # next mic chunk starts a new recording


def log_audio(role: str, wav: bytes) -> str:
    """Queue a WAV for saving; returns its path relative to the session folder."""
    with log_lock:
        session["n"][role] += 1
        rel = f"audio/{role}/{session['n'][role]:04d}.wav"
        log_q.put(("file", session["dir"] / rel, wav))
    return rel


def log_chat(role: str, text: str, audio: str | None = None, **extra):
    with log_lock:
        entry = {"time": datetime.now().isoformat(timespec="milliseconds"), "mode": session["kind"],
                 "role": role, "text": text, "audio": audio, **extra}
        log_q.put(("chat", session["dir"], entry))


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
        asr({"raw": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
            generate_kwargs={"language": "ja", "task": "transcribe"})

    def tts_step():
        model_id = os.environ.get("TTS_MODEL") or list_tts_models()[0]["id"]
        synth(TTSRequest(text="こんにちは。今日はいい天気ですね。", model=model_id))

    def filler_step():  # cache the backchannel phrases for the default voice so they play instantly
        model_id = os.environ.get("TTS_MODEL") or list_tts_models()[0]["id"]
        for text in FILLERS:
            req = TTSRequest(text=text, model=model_id)
            _filler_cache[(req.model, req.style, req.style_weight, req.length, text)] = synth(req)

    def llm_step():
        for _ in client.models.generate_content_stream(model=LLM_MODEL, contents="こんにちは", config=LLM_CONFIG):
            pass

    step("ASR", asr_step)
    step("TTS", tts_step)
    step("fillers", filler_step)
    step("LLM", llm_step)


@asynccontextmanager
async def lifespan(_app):
    await asyncio.to_thread(warmup)  # the server starts accepting requests only after this
    yield
    with log_lock:
        finalize_recording_locked()
    log_q.put(None)  # flush pending log writes
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


@app.post("/api/asr")
def api_asr(audio: UploadFile = File(...), kind: str = Form("user"), offset: float | None = Form(None)):
    # kind: "user" | "backchannel"; offset: where this utterance starts in the continuous mic recording (seconds)
    t0 = time.perf_counter()
    raw = audio.file.read()
    samples = decode_audio(raw)
    if len(samples) < 1600:  # < 0.1 s
        return {"text": ""}
    result = asr({"raw": samples, "sampling_rate": 16000}, generate_kwargs={"language": "ja", "task": "transcribe"})
    text = result["text"].strip()
    log_time("ASR", t0, f"(audio {len(samples) / 16000:.1f}s) -> {text!r}")
    if text:
        with log_lock:
            full = mic_rec["rel"]
        log_chat(kind if kind in ("user", "backchannel") else "user", text, log_audio("human", raw),
                 full_audio=full if offset is not None else None,
                 full_offset_s=round(offset, 3) if offset is not None and full else None,
                 duration_s=round(len(samples) / 16000, 3))
    return {"text": text}


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
    reply = client.models.generate_content(model=LLM_MODEL, contents=history + [user], config=LLM_CONFIG).text.strip()
    history.extend([user, turn("model", reply)])
    log_time("LLM", t0, f"-> {reply!r}")
    return {"reply": reply}


def synth(req: TTSRequest) -> bytes:
    """Text -> WAV bytes."""
    model = get_tts(req.model)
    style = req.style if req.style in model.style2id else next(iter(model.style2id))
    # SBV2's g2p crashes on "!" before more text ("Input must be katakana only: ！"), so use "。" instead.
    text = req.text.replace("！", "。").replace("!", "。")
    sr, wav = model.infer(
        text=text, language=Languages.JP, style=style, style_weight=req.style_weight, length=req.length,
    )
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


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


SENTENCE_END = re.compile(r"[^。！？!?\n]*[。！？!?\n]+")
SPEAKABLE = re.compile(r"[^\W_]")  # at least one letter/kana/kanji/digit


class StreamRequest(TTSRequest):
    test: bool = False  # debug: speak random canned sentences instead of calling the LLM
    # `text` is the user's message; the rest are TTS options


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
    if not req.text.strip() and not req.test:
        raise HTTPException(400, "Empty text")

    def events():
        global turn_no, turn_done
        with turn_lock:
            turn_no += 1
            my_turn, prev_done, turn_done = turn_no, turn_done, threading.Event()
            done = turn_done
        prev_done.wait(timeout=5)  # let the cancelled reply commit its history first

        cancelled = lambda: my_turn != turn_no
        t0 = time.perf_counter()
        user = turn("user", req.text.strip())
        spoken, buf, first = [], "", True

        def speak(sentence):
            sentence = sentence.strip()
            if not SPEAKABLE.search(sentence):
                return None
            t1 = time.perf_counter()
            audio = synth(req.model_copy(update={"text": sentence}))
            log_time("TTS", t1, f"-> {sentence!r} (since start {time.perf_counter() - t0:.2f}s)")
            spoken.append(sentence)
            clip_id = log_audio("ai", audio)
            log_chat("ai", sentence, clip_id, turn=my_turn)
            return json.dumps({"text": sentence, "audio": base64.b64encode(audio).decode(), "id": clip_id}) + "\n"

        try:
            chunks = (
                [SimpleNamespace(text=random.choice(TEST_REPLIES))] if req.test
                else client.models.generate_content_stream(model=LLM_MODEL, contents=history + [user], config=LLM_CONFIG)
            )
            for chunk in chunks:
                if cancelled():
                    return
                if not chunk.text:
                    continue
                if first:
                    log_time("LLM first chunk", t0)
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
            log_time("LLM+TTS total", t0)
        finally:
            if cancelled():
                print(f"[interrupted] after {len(spoken)} sentence(s)", flush=True)
                log_chat("event", "interrupted", sentences_sent=len(spoken), turn=my_turn)
            if spoken and not req.test:
                history.extend([user, turn("model", "".join(spoken))])
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
