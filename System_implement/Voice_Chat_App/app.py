"""
Voice chat web app: browser mic -> Kotoba-Whisper (ASR) -> Response_weight_random (LLM)
-> Style-Bert-VITS2 (TTS, model selectable in the UI).

Run:  python app.py   (then open http://localhost:8000)
Needs GENAI_API_KEY in the environment (used by Response_weight_random).
"""

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
SYSTEM_DIR = HERE.parent
SBV2_DIR = SYSTEM_DIR / "Style-Bert-VITS2"
MODEL_ASSETS = Path(os.environ.get("SBV2_MODEL_ASSETS", SBV2_DIR / "model_assets"))

# Use the local Style-Bert-VITS2 checkout's package and the chatbot module.
sys.path.insert(0, str(SBV2_DIR))
sys.path.insert(0, str(SYSTEM_DIR / "LLM_Response" / "src"))

import Response_weight_random as chatbot  # noqa: E402
from style_bert_vits2.constants import Languages  # noqa: E402
from style_bert_vits2.nlp import bert_models  # noqa: E402
from style_bert_vits2.tts_model import TTSModel  # noqa: E402

ASR_MODEL_ID = os.environ.get("ASR_MODEL", "kotoba-tech/kotoba-whisper-v2.0")
SBV2_BERT_ID = "ku-nlp/deberta-v2-large-japanese-char-wwm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LOADED_TTS_MODELS = 2

@asynccontextmanager
async def lifespan(_app):
    # Load every model and run a dummy inference before the server accepts requests,
    # so the first real turn doesn't pay load / CUDA-warmup cost.
    await asyncio.to_thread(warmup)
    yield


app = FastAPI(title="Voice Chat", lifespan=lifespan)
_gpu_lock = threading.Lock()  # ASR / TTS / LLM share one GPU; keep requests serial


# ---------------- ASR ----------------
_asr = None


def get_asr():
    global _asr
    if _asr is None:
        from transformers import pipeline
        _asr = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_ID,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device=DEVICE,
            chunk_length_s=15,
        )
    return _asr


def decode_audio(data: bytes) -> np.ndarray:
    """Any browser-recorded container (webm/ogg/mp4/wav) -> 16 kHz mono float32 via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
        input=data, capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(400, f"ffmpeg could not decode audio: {proc.stderr.decode(errors='ignore')[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


# ---------------- TTS (Style-Bert-VITS2) ----------------
_bert_loaded = False
_tts_cache: dict[str, TTSModel] = {}  # insertion-ordered; oldest evicted first


def list_tts_models() -> list[dict]:
    """One entry per .safetensors checkpoint under model_assets/<dir>/ that has config.json + style_vectors.npy."""
    models = []
    for d in sorted(p for p in MODEL_ASSETS.iterdir() if p.is_dir()):
        config, style_vec = d / "config.json", d / "style_vectors.npy"
        if not (config.exists() and style_vec.exists()):
            continue
        styles = list(json.loads(config.read_text(encoding="utf-8"))["data"].get("style2id", {"Neutral": 0}))
        for ckpt in sorted(d.glob("*.safetensors")):
            models.append({"id": f"{d.name}/{ckpt.name}", "name": d.name, "checkpoint": ckpt.name, "styles": styles})
    return models


def get_tts(model_id: str) -> TTSModel:
    global _bert_loaded
    if not _bert_loaded:
        # transformers>=5 loads in the checkpoint's dtype (fp16 here); SBV2's net expects fp32.
        bert_models.load_model(Languages.JP, SBV2_BERT_ID).float()
        bert_models.load_tokenizer(Languages.JP, SBV2_BERT_ID)
        _bert_loaded = True
    if model_id in _tts_cache:
        return _tts_cache[model_id]

    folder, _, ckpt = model_id.partition("/")
    model_dir = (MODEL_ASSETS / folder).resolve()
    ckpt_path = (model_dir / ckpt).resolve()
    if MODEL_ASSETS.resolve() not in model_dir.parents or model_dir not in ckpt_path.parents or not ckpt_path.exists():
        raise HTTPException(404, f"Unknown TTS model: {model_id}")

    while len(_tts_cache) >= MAX_LOADED_TTS_MODELS:
        del _tts_cache[next(iter(_tts_cache))]
    torch.cuda.empty_cache()
    _tts_cache[model_id] = TTSModel(
        model_path=ckpt_path,
        config_path=model_dir / "config.json",
        style_vec_path=model_dir / "style_vectors.npy",
        device=DEVICE,
    )
    return _tts_cache[model_id]


# ---------------- Chat state ----------------
def new_chat_state():
    return {"summary": "", "recent": [], "opponent_style": "neutral", "pronoun_choice": None, "koshou_choice": None}


chat_state = new_chat_state()  # single-user app: one conversation


# ---------------- API ----------------
class ChatRequest(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    model: str
    style: str = "Neutral"
    style_weight: float = 5.0
    length: float = 1.0  # >1.0 slower


# ---------------- Warm-up ----------------
def warmup():
    """Each step is independent and non-fatal: a failure just means that component loads lazily later."""
    def step(name, fn):
        t0 = time.perf_counter()
        try:
            fn()
            print(f"[warmup] {name} ready ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            print(f"[warmup] {name} skipped: {e!r}")

    def asr():
        with _gpu_lock:
            get_asr()({"raw": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000},
                      generate_kwargs={"language": "ja", "task": "transcribe"})

    def tts():
        models = list_tts_models()
        model_id = os.environ.get("TTS_MODEL") or models[0]["id"]
        with _gpu_lock:
            get_tts(model_id).infer(text="こんにちは", language=Languages.JP)

    def llm():
        chatbot.get_tagger()("こんにちは")
        chatbot.get_client().models.generate_content(model=chatbot.MODEL, contents="こんにちは")

    step("ASR", asr)
    step("TTS", tts)
    step("LLM", llm)


@app.get("/api/models")
def api_models():
    return list_tts_models()


@app.post("/api/asr")
def api_asr(audio: UploadFile = File(...)):
    samples = decode_audio(audio.file.read())
    if len(samples) < 1600:  # < 0.1 s
        return {"text": ""}
    with _gpu_lock:
        result = get_asr()(
            {"raw": samples, "sampling_rate": 16000},
            generate_kwargs={"language": "ja", "task": "transcribe"},
        )
    return {"text": result["text"].strip()}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not req.text.strip():
        raise HTTPException(400, "Empty text")
    with _gpu_lock:
        reply = chatbot.generate_response(chat_state, req.text.strip())
    return {"reply": reply}


@app.post("/api/reset")
def api_reset():
    global chat_state
    chat_state = new_chat_state()
    return {"ok": True}


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    with _gpu_lock:
        model = get_tts(req.model)
        style = req.style if req.style in model.style2id else next(iter(model.style2id))
        sr, audio = model.infer(
            text=req.text,
            language=Languages.JP,
            style=style,
            style_weight=req.style_weight,
            length=req.length,
        )
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
