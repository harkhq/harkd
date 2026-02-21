"""FastAPI worker app for remote GPU transcription.

Exposes /transcribe and /health endpoints.
Deployed as a Docker container on GPU cloud providers.
"""

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile

import httpx
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="harkd-worker")

WORKER_API_KEY = os.environ.get("HARKD_WORKER_API_KEY", "")


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Verify worker API key for all endpoints except /health.

    Accepts the key in either:
    - Authorization: Bearer <key>  (standard, used by Koyeb/Scaleway)
    - X-Worker-Api-Key: <key>      (used by Verda, where Authorization
                                     carries the provider's inference key)
    """
    if request.url.path == "/health":
        return await call_next(request)

    if not WORKER_API_KEY:
        return JSONResponse(
            status_code=403,
            content={"error": "No API key configured on worker"},
        )

    # Check Authorization header first, then fall back to X-Worker-Api-Key
    auth = request.headers.get("Authorization", "")
    alt_key = request.headers.get("X-Worker-Api-Key", "")
    if auth == f"Bearer {WORKER_API_KEY}" or alt_key == WORKER_API_KEY:
        return await call_next(request)

    return JSONResponse(
        status_code=401,
        content={"error": "Invalid or missing API key"},
    )


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile | None = None,
    audio_url: str | None = Form(None),
    params: str = Form(...),
):
    """Accept audio (file upload OR URL) + JSON params, return transcription result."""
    params_dict = json.loads(params)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            if audio:
                shutil.copyfileobj(audio.file, tmp)
            elif audio_url:
                # Stream download to avoid loading entire file into memory
                async with httpx.AsyncClient() as client, client.stream("GET", audio_url) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        tmp.write(chunk)
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Either audio file or audio_url is required"},
                )

        from harkd.services.transcription_worker import transcribe_audio_worker

        # Run in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            transcribe_audio_worker,
            audio_path_str=tmp_path,
            model_name=params_dict.get("model_name", "large-v3"),
            language=params_dict.get("language"),
            word_timestamps=params_dict.get("word_timestamps", False),
            diarize=params_dict.get("diarize", False),
            hf_token=params_dict.get("hf_token"),
            beam_size=params_dict.get("beam_size", 3),
            batch_size=params_dict.get("batch_size", 16),
            vad_onset=params_dict.get("vad_onset", 0.5),
            vad_offset=params_dict.get("vad_offset", 0.363),
            vad_method=params_dict.get("vad_method", "pyannote"),
            num_speakers=params_dict.get("num_speakers"),
            min_speakers=params_dict.get("min_speakers"),
            max_speakers=params_dict.get("max_speakers"),
            clustering_threshold=params_dict.get("clustering_threshold"),
        )
        return JSONResponse(result)
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


@app.get("/health")
async def health():
    """Health check reporting GPU and model status."""
    import torch

    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
