"""FastAPI service exposing the Bengali whisper-bn transcription queue."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional

from .audio import AudioPreprocessor
from .pipeline import TranscriptionPipeline
from .queue import TranscriptionQueue

try:
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
    from pydantic import BaseModel, Field
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
except ImportError as exc:  # pragma: no cover
    raise SystemExit("FastAPI dependencies are missing. Install with: pip install '.[api]'") from exc

APP_NAME = "whisper-bn"
PUBLIC_JOB_FIELDS = (
    "id",
    "status",
    "engine",
    "language",
    "labels",
    "transcript",
    "result",
    "error",
    "created_at",
    "updated_at",
)
UPLOAD_DIR = Path(os.getenv("ASR_UPLOAD_DIR", "./transcripts/uploads"))
OUTPUT_DIR = Path(os.getenv("ASR_OUTPUT_DIR", "./transcripts"))
DB_PATH = os.getenv("ASR_QUEUE_DB", str(OUTPUT_DIR / "transcription_queue.sqlite3"))

app = FastAPI(
    title="whisper-bn transcription API",
    version="1.0.0",
    description=(
        "Public Bengali speech-to-text API backed by the SAM15K whisper-bn engine. "
        "Upload one audio file, receive a queued transcription job, then poll the job "
        "endpoint until it completes. Completed job responses include the transcript "
        "JSON inline under `transcript` and `result`. Public job responses do not "
        "include server filesystem paths or artifact URLs. "
        "Only the `whisper-bn` engine is exposed."
    ),
)
queue = TranscriptionQueue(DB_PATH)
_JOB_GEMINI_KEYS: dict[str, str] = {}


def _normalize_engine(engine: str) -> str:
    selected = (engine or "whisper-bn").strip().lower()
    if selected in {"whisper-bn", "local", "local-openai", "whisper-1"}:
        return "whisper-bn"
    if selected == "gemini":
        return "gemini"
    raise HTTPException(status_code=400, detail="Unsupported transcription engine")


def _require_bearer(authorization: Optional[str]) -> None:
    expected = os.getenv("SITE_API_KEY")
    if not expected:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.middleware("http")
async def require_bearer_for_v1(request: Request, call_next):
    if request.url.path.startswith("/v1/"):
        try:
            _require_bearer(request.headers.get("authorization"))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.get(
    "/health",
    summary="Health",
    description="Returns service readiness and the public engine name.",
    responses={
        200: {
            "description": "The API process is running and ready to accept transcription jobs.",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "engine": "whisper-bn"}
                }
            },
        }
    },
)
def health() -> dict:
    return {"status": "ok", "engine": APP_NAME}


@app.post(
    "/v1/audio/transcriptions",
    summary="Create OpenAI-compatible transcription",
    description="Transcribe audio with the OpenAI-compatible audio transcription shape. Set async=true to receive a queued job id for long audio.",
)
async def create_openai_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form("bn"),
    diarize: bool = Form(False),
    async_response: bool = Form(False, alias="async"),
    engine: str = Form("whisper-bn"),
    gemini_api_key: Optional[str] = Form(None),
    labels: str = Form("Agent,Customer"),
) -> JSONResponse:
    job = _create_uploaded_job(file, language, diarize, engine, model, gemini_api_key, labels)
    if async_response:
        background_tasks.add_task(_drain_queue)
        return JSONResponse(status_code=202, content={"job_id": job.id, "status": job.status, "engine": job.engine})

    try:
        result_path = await _process_job(job)
        queue.complete_job(job.id, result_path)
        transcript = _read_json(Path(result_path))
        return JSONResponse(content=_openai_transcription_response(transcript, job))
    except Exception as exc:  # noqa: BLE001
        queue.fail_job(job.id, str(exc))
        _cleanup_audio_artifacts(job.audio_path, Path(job.output_dir) / "_temp")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/v1/transcriptions/{job_id}",
    summary="Get Transcription",
    description=(
        "Return the queued job state. When complete, the response includes the full "
        "transcript JSON inline under `transcript` and `result`. Public responses "
        "do not expose server filesystem paths or artifact URLs."
    ),
    responses={
        200: {
            "description": "Current transcription job state.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "01J4W4Q0R7M7MZ3M8P0N9B4K2T",
                        "status": "completed",
                        "engine": "whisper-bn",
                        "language": "bn",
                        "labels": "Agent,Customer",
                        "transcript": {
                            "call_id": "example",
                            "filename": "example.wav",
                            "language_detected": "bn",
                            "segments": [
                                {"start": 0.0, "end": 2.4, "speaker": "Agent", "text": "হ্যালো"}
                            ],
                            "full_text": "[Agent]: হ্যালো",
                            "status": "success"
                        },
                        "error": None,
                        "created_at": "2026-07-22 11:10:00",
                        "updated_at": "2026-07-22 11:11:30",
                    }
                }
            },
        },
        404: {
            "description": "No transcription job exists for the supplied job_id.",
            "content": {"application/json": {"example": {"detail": "Job not found"}}},
        },
    },
)
def get_transcription(job_id: str, request: Request) -> dict:
    job = _get_job_or_404(job_id)
    return _job_to_http_dict(job, request)


@app.get(
    "/v1/transcriptions",
    summary="List Transcriptions",
    description="List recent transcription jobs, newest first, for integration dashboards or polling tools.",
    responses={
        200: {
            "description": "Recent transcription jobs.",
            "content": {
                "application/json": {
                    "example": {
                        "jobs": [
                            {
                                "id": "01J4W4Q0R7M7MZ3M8P0N9B4K2T",
                                "status": "completed",
                                "engine": "whisper-bn",
                                "language": "bn",
                                "labels": "Agent,Customer",
                                "transcript": {"call_id": "example", "status": "success", "full_text": "[Agent]: হ্যালো"},
                                "error": None,
                            }
                        ]
                    }
                }
            },
        }
    },
)
def list_transcriptions(request: Request, limit: int = 100) -> dict:
    return {"jobs": [_job_to_http_dict(job, request) for job in queue.list_jobs(limit=limit)]}


class AIProcessRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=120_000)
    prompt: str = Field(min_length=1, max_length=12_000)
    openai_api_key: str = Field(min_length=8, max_length=512)
    model: str = Field(min_length=1, max_length=128)
    base_url: str = Field(default="https://api.openai.com/v1", min_length=8, max_length=2048)
    provider: str = Field(default="openai", max_length=32)


@app.post("/v1/ai-process", summary="Process a transcript with a user-supplied OpenAI key")
def ai_process(request_body: AIProcessRequest) -> dict:
    """Relay one transcript-processing request without storing the caller's provider key."""
    provider = (request_body.provider or "openai").strip().lower()
    if provider == "gemini":
        return _gemini_process(request_body)
    if provider != "openai":
        raise HTTPException(status_code=400, detail="Unsupported AI provider")
    return _openai_process(request_body)


def _openai_process(request_body: AIProcessRequest) -> dict:
    key = request_body.openai_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Enter an API key")
    model = request_body.model.strip()
    parsed_url = urlparse(request_body.base_url.strip())
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise HTTPException(status_code=400, detail="Enter an HTTPS OpenAI-compatible base URL")
    base_url = request_body.base_url.strip().rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Follow the user's instruction. Preserve Bengali faithfully. Do not invent facts."},
            {"role": "user", "content": f"Instruction:\n{request_body.prompt.strip()}\n\nTranscript:\n{request_body.transcript.strip()}"},
        ],
        "temperature": 0.2,
    }).encode("utf-8")
    upstream = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(upstream, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("OpenAI returned no content")
        return {"output": content, "model": model}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code in (401, 403):
            raise HTTPException(status_code=401, detail="OpenAI rejected the supplied API key") from exc
        raise HTTPException(status_code=502, detail=f"OpenAI request failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="OpenAI processing failed") from exc


def _gemini_process(request_body: AIProcessRequest) -> dict:
    key = request_body.openai_api_key.strip()
    model = request_body.model.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Enter a Gemini API key")
    try:
        from google import genai
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=f"Follow the user's instruction. Preserve Bengali faithfully. Do not invent facts.\n\nInstruction:\n{request_body.prompt.strip()}\n\nTranscript:\n{request_body.transcript.strip()}",
        )
        content = response.text or ""
        if not content:
            raise ValueError("Gemini returned no content")
        return {"output": content, "model": model}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="Gemini processing failed") from exc


@app.get(
    "/v1/transcriptions/{job_id}/result",
    summary="Download Transcript JSON",
    description=(
        "Return the completed transcript JSON over HTTP. The normal job polling "
        "response also includes this same transcript inline under `transcript` and `result`."
    ),
    responses={
        200: {
            "description": "Completed transcript JSON.",
            "content": {"application/json": {"example": {"call_id": "call", "status": "success", "full_text": "[Agent]: হ্যালো"}}},
        },
        404: {"description": "Job or transcript artifact not found."},
        409: {"description": "Job is not completed yet."},
    },
)
def get_transcription_result(job_id: str) -> JSONResponse:
    path = _completed_artifact_path(job_id, ".json")
    return JSONResponse(content=_read_json(path))


@app.get(
    "/v1/transcriptions/{job_id}/text",
    summary="Download Transcript Text",
    description="Return the completed human-readable transcript text over HTTP.",
    responses={
        200: {
            "description": "Plain text transcript.",
            "content": {"text/plain": {"example": "[Agent]: হ্যালো"}},
        },
        404: {"description": "Job or transcript artifact not found."},
        409: {"description": "Job is not completed yet."},
    },
)
def get_transcription_text(job_id: str) -> PlainTextResponse:
    json_path = _completed_artifact_path(job_id, ".json")
    text_path = json_path.with_suffix(".txt")
    if text_path.exists():
        return PlainTextResponse(text_path.read_text(encoding="utf-8"))
    transcript = _read_json(json_path)
    return PlainTextResponse(str(transcript.get("full_text") or ""))


@app.get(
    "/v1/transcriptions/{job_id}/srt",
    summary="Download Transcript SRT",
    description="Return the completed SRT subtitle artifact over HTTP when available.",
    responses={
        200: {"description": "SRT subtitle file.", "content": {"application/x-subrip": {}}},
        404: {"description": "Job or SRT artifact not found."},
        409: {"description": "Job is not completed yet."},
    },
)
def get_transcription_srt(job_id: str) -> FileResponse:
    path = _completed_artifact_path(job_id, ".srt")
    return FileResponse(path, media_type="application/x-subrip", filename=path.name)


def _get_job_or_404(job_id: str):
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _job_to_http_dict(job, request: Request | None) -> dict:
    internal = TranscriptionQueue.job_to_dict(job)
    public = {field: internal.get(field) for field in PUBLIC_JOB_FIELDS}
    public["engine"] = public.get("engine") or APP_NAME
    return {key: value for key, value in public.items() if value is not None}


def _completed_artifact_path(job_id: str, suffix: str) -> Path:
    job = _get_job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Transcription is not completed yet")
    if not job.result_json_path:
        raise HTTPException(status_code=404, detail="Transcript artifact not found")
    path = Path(job.result_json_path).with_suffix(suffix)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Transcript artifact not found: {suffix}")
    return path


def _read_json(path: Path) -> dict:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Transcript JSON artifact is invalid") from exc


def _create_uploaded_job(
    file: UploadFile,
    language: Optional[str],
    diarize: bool,
    engine: str,
    model: Optional[str],
    gemini_api_key: Optional[str],
    labels: str,
):
    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix and suffix not in AudioPreprocessor.SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {suffix}")
    engine = _normalize_engine(engine)
    model = (model or "").strip() or None
    gemini_api_key = (gemini_api_key or "").strip() or None
    if engine == "gemini" and not (gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEYS")):
        raise HTTPException(status_code=400, detail="Enter a Gemini API key")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "audio.wav").name
    path = UPLOAD_DIR / f"{os.urandom(8).hex()}_{safe_name}"
    with path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job = queue.create_job(
        audio_path=str(path),
        output_dir=str(OUTPUT_DIR),
        language=None if language in (None, "", "auto") else language,
        labels=labels,
        diarize=diarize,
        engine=engine,
        model=model,
    )
    if gemini_api_key:
        _JOB_GEMINI_KEYS[job.id] = gemini_api_key
    return job


def _openai_transcription_response(transcript: dict, job) -> dict:
    segments = transcript.get("segments") or transcript.get("turns") or []
    return {
        "text": transcript.get("full_text") or "\n".join(str(segment.get("text", "")) for segment in segments).strip(),
        "language": transcript.get("language_detected") or job.language or "bn",
        "duration": transcript.get("duration_seconds"),
        "model": transcript.get("model_used") or job.model or job.engine,
        "segments": [
            {
                "id": index,
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", ""),
                "speaker": segment.get("speaker") or "",
            }
            for index, segment in enumerate(segments)
        ],
    }


async def _process_job(job) -> str:
    labels = tuple((job.labels.split(",", 1) + ["Customer"])[:2])
    engine = _normalize_engine(job.engine)
    pipeline = TranscriptionPipeline(
        output_dir=job.output_dir,
        language=job.language,
        speaker_labels=labels,
        engine=engine,
        whisper_model_id=os.getenv("WHISPER_MODEL") if engine == "whisper-bn" else None,
        google_api_key=_JOB_GEMINI_KEYS.pop(job.id, None) if engine == "gemini" else None,
        gemini_model_id=job.model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    )
    if job.diarize and engine != "gemini":
        from .diarization import transcribe_diarized
        started = asyncio.get_running_loop().time()
        transcript = await asyncio.to_thread(transcribe_diarized, job.audio_path, pipeline.transcriber, job.language, pipeline.temp_dir, pipeline._model_label)
        transcript.processing_time_seconds = round(asyncio.get_running_loop().time() - started, 2)
        pipeline._save_outputs(transcript)
    else:
        transcript = await asyncio.to_thread(pipeline.process_file, job.audio_path, False)
    result_path = str(Path(job.output_dir) / f"{transcript.call_id}.json")
    if not Path(result_path).exists():
        candidates = sorted(Path(job.output_dir).glob(f"{transcript.call_id}*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            result_path = str(candidates[0])
        else:
            raise RuntimeError("Transcript artifact was not written")
    if transcript.status != "success":
        raise RuntimeError(transcript.error or "Transcription failed")
    _cleanup_audio_artifacts(job.audio_path, pipeline.temp_dir)
    return result_path


def _cleanup_audio_artifacts(audio_path: str, temp_dir: Path) -> None:
    """Delete uploaded audio and only its derived temporary files after a terminal job."""
    source = Path(audio_path)
    try:
        source.unlink(missing_ok=True)
    except OSError:
        pass
    stem = source.stem
    for path in temp_dir.glob(f"{stem}*"):
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
    derived_dir = temp_dir / f"diarized_{stem}"
    if derived_dir.is_dir():
        shutil.rmtree(derived_dir, ignore_errors=True)


async def _drain_queue() -> None:
    while True:
        job = queue.claim_next()
        if not job:
            return
        try:
            result_path = await _process_job(job)
            queue.complete_job(job.id, result_path)
        except Exception as exc:  # noqa: BLE001
            queue.fail_job(job.id, str(exc))
            _cleanup_audio_artifacts(job.audio_path, Path(job.output_dir) / "_temp")
