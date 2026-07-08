import base64
import io
import time
import re
import numpy as np
import soundfile as sf
import asyncio
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..structures.schemas import (
    OpenAISpeechRequest,
    ModelInfo,
    VoiceInfo,
    CreateVoiceRequest,
)
from ..services.audio_encoding import encode_audio, get_content_type
from ..services.text_processing import normalize_text
from ..services import voice_manager

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)

AVAILABLE_MODELS = [
    ModelInfo(
        id="omnivoice",
        object="model",
        created=int(time.time()),
        owned_by="omnivoice",
    )
]

import os
import uuid
from ..config import settings

VOICE_LIBRARY = [
    VoiceInfo(voice_id="default", name="Default Voice", language="Auto"),
    VoiceInfo(voice_id="clone:Example", name="Example Clone", language="Auto", description="Example voice clone profile format")
]

import asyncio

_generation_semaphore: Optional[asyncio.Semaphore] = None

def get_generation_semaphore() -> asyncio.Semaphore:
    global _generation_semaphore
    if _generation_semaphore is None:
        _generation_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GENERATIONS)
    return _generation_semaphore

def _pieces(text: str, max_chars: int) -> List[str]:
    """Break text into pieces each <= max_chars, splitting on sentence
    punctuation (. ! ?), then clause punctuation (, ; :), then words."""
    out: List[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        for clause in re.split(r"(?<=[,;:])\s+", sent):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= max_chars:
                out.append(clause)
                continue
            buf = ""
            for w in clause.split():
                if not buf:
                    buf = w
                elif len(buf) + 1 + len(w) <= max_chars:
                    buf += " " + w
                else:
                    out.append(buf)
                    buf = w
            if buf:
                out.append(buf)
    return out

def _split_into_chunks(text: str, min_chars: int, max_chars: int) -> List[str]:
    """Split text into chunks within a [min_chars, max_chars] window, breaking
    at punctuation. Pieces are packed greedily up to max_chars; a chunk shorter
    than min_chars is merged into a neighbour when the result still fits."""
    pieces = _pieces(text, max_chars)
    if not pieces:
        return []
    # Greedy pack up to max_chars.
    chunks: List[str] = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += " " + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    # Soft minimum: fold an undersized chunk into a neighbour if it still fits.
    merged: List[str] = []
    for c in chunks:
        if (
            merged
            and (len(c) < min_chars or len(merged[-1]) < min_chars)
            and len(merged[-1]) + 1 + len(c) <= max_chars
        ):
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]

# Lazy loaded backend instance
_backend_instance = None

def get_backend():
    global _backend_instance
    if _backend_instance is None:
        from ..backends.omnivoice_backend import OmniVoiceStreamModel
        model_path = "k2-fsa/OmniVoice" 
        try:
            print(f"Loading OmniVoice backend from {model_path}...")
            _backend_instance = OmniVoiceStreamModel.from_pretrained(model_path)
            print("OmniVoice backend loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load actual model. Error: {e}")
            raise RuntimeError(f"Failed to load OmniVoice model: {e}")
    return _backend_instance

async def generate_speech(
    text: str, 
    language: str, 
    instruct: str,
    voice_clone_prompt=None
) -> tuple[np.ndarray, int]:
    """Generate speech handling chunking at the router level like Qwen3."""
    backend = get_backend()
    
    async def _synth(segment: str) -> tuple[np.ndarray, int]:
        def run_inference():
            return backend.generate_voice_clone(
                text=segment,
                language=language,
                instruct=instruct,
                voice_clone_prompt=voice_clone_prompt
            )
        # Выносим блокирующий вызов модели в отдельный поток
        audios, sr = await asyncio.to_thread(run_inference)
        return audios[0], sr

    chunks = (
        _split_into_chunks(text, settings.MIN_CHUNK_CHARS, settings.MAX_CHUNK_CHARS)
        if settings.AUTOCHUNK else [text]
    )
    
    if len(chunks) <= 1:
        return await _synth(text)

    # Параллельная генерация аудио для всех чанков
    tasks = [_synth(seg) for seg in chunks]
    results = await asyncio.gather(*tasks)

    audios: List[np.ndarray] = []
    sr = settings.DEFAULT_SAMPLE_RATE
    
    for a, chunk_sr in results:
        sr = chunk_sr
        if a is not None and len(a):
            audios.append(np.asarray(a))
            
    if not audios:
        raise RuntimeError("No audio produced from any chunk")
        
    gap_len = int(sr * settings.CHUNK_GAP_MS / 1000.0)
    gap = np.zeros(gap_len, dtype=audios[0].dtype) if gap_len > 0 else None
    
    merged: List[np.ndarray] = []
    for i, a in enumerate(audios):
        if i > 0 and gap is not None:
            merged.append(gap)
        merged.append(a)
        
    return np.concatenate(merged), sr

@router.get("/audio/voices")
async def list_voices() -> list:
    voices_from_db = await voice_manager.get_all_voices_metadata()
    static_voices = [v.dict() for v in VOICE_LIBRARY]
    return static_voices + voices_from_db

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print("Loading STT model...")
            _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        except ImportError:
            raise RuntimeError("faster-whisper is not installed. Please install it to use STT automation.")
    return _whisper_model

@router.post("/audio/voices")
async def add_voice(request: CreateVoiceRequest):
    sem = get_generation_semaphore()
    if sem.locked():
        raise HTTPException(status_code=429, detail="Too Many Requests: Server is currently at maximum capacity (Local).")
        
    try:
        backend = get_backend()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    try:
        if request.ref_audio.startswith("http://") or request.ref_audio.startswith("https://"):
            import urllib.request
            def download_audio():
                req = urllib.request.Request(request.ref_audio, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    return response.read()
            audio_bytes = await asyncio.to_thread(download_audio)
        else:
            audio_bytes = base64.b64decode(request.ref_audio)
            
        audio_io = io.BytesIO(audio_bytes)
        ref_audio_np, ref_sr = sf.read(audio_io)
        if len(ref_audio_np.shape) > 1:
            ref_audio_np = ref_audio_np.mean(axis=1) 
        import torch
        ref_audio_tensor = torch.from_numpy(ref_audio_np).float()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process reference audio: {e}")

    ref_text = request.ref_text
    if not ref_text or not ref_text.strip():
        try:
            whisper_model = get_whisper_model()
            audio_io.seek(0)
            def run_stt():
                segments, info = whisper_model.transcribe(audio_io, beam_size=5)
                return " ".join([segment.text for segment in segments])
            ref_text = await asyncio.to_thread(run_stt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"STT failed: {e}")

    voice_id = str(uuid.uuid4())

    try:
        def run_create_prompt():
            return backend.create_voice_clone_prompt(
                ref_audio=(ref_audio_tensor, ref_sr),
                ref_text=ref_text,
                preprocess_prompt=True
            )
            
        async with sem:
            prompt_obj = await asyncio.to_thread(run_create_prompt)
        
        await voice_manager.save_voice(
            voice_id=voice_id,
            name=request.name,
            language=request.language,
            prompt_obj=prompt_obj,
            description=request.description
        )
        
        return {"status": "success", "voice_id": voice_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create voice: {e}")

@router.delete("/audio/voices/{voice_id}")
async def remove_voice(voice_id: str):
    # Protect default voices
    if voice_id == "default" or voice_id.startswith("clone:"):
        raise HTTPException(status_code=400, detail="Cannot delete default voices.")
        
    deleted = await voice_manager.delete_voice(voice_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found.")
        
    return {"status": "success", "message": f"Voice '{voice_id}' deleted."}

@router.post("/audio/speech")
async def create_speech(request: OpenAISpeechRequest):
    sem = get_generation_semaphore()
    if sem.locked():
        raise HTTPException(status_code=429, detail="Too Many Requests: Server is currently at maximum capacity (Local).")

    try:
        backend = get_backend()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    normalized_text = normalize_text(request.input, request.normalization_options, request.language)
    if not normalized_text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty after normalization")

    voice_clone_prompt = None
    if request.voice_id and request.voice_id.lower() != "default":
        if not request.voice_id.startswith("clone:"):
            voice_clone_prompt = await voice_manager.get_voice(request.voice_id)
            if voice_clone_prompt is None:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice_id}' not found.")

    if request.stream:
        if request.response_format not in ("pcm", "wav"):
            raise HTTPException(status_code=400, detail="Streaming only supports pcm/wav")
        
        async def audio_generator():
            async with sem:
                generator = backend.stream_generate_voice_clone(
                    text=normalized_text,
                    language=request.language,
                    instruct=request.instruct,
                    voice_clone_prompt=voice_clone_prompt,
                    audio_chunk_duration=3.0
                )
                for chunk, sr in generator:
                    yield encode_audio(chunk, sr, request.response_format)
                
        return StreamingResponse(
            audio_generator(),
            media_type=get_content_type(request.response_format)
        )

    try:
        async with sem:
            audio, sr = await generate_speech(
                text=normalized_text,
                language=request.language,
                instruct=request.instruct,
                voice_clone_prompt=voice_clone_prompt
            )
        
        encoded_audio = encode_audio(audio, sr, request.response_format)
        
        return Response(
            content=encoded_audio,
            media_type=get_content_type(request.response_format)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


