"""Local Pyannote diarization followed by per-turn Bengali ASR."""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _pipeline():
    from pyannote.audio import Inference, Model, Pipeline

    model_dir = Path(os.getenv("PYANNOTE_MODEL_DIR", "/models/pyannote/community-1"))
    if not model_dir.is_dir():
        raise RuntimeError("Local Pyannote Community-1 model is not installed")
    pipeline = Pipeline.from_pretrained(model_dir)
    checkpoint = Path(os.getenv("PYANNOTE_SEGMENTATION_CHECKPOINT", ""))
    if checkpoint.is_file():
        finetuned = Model.from_pretrained(checkpoint, strict=False)
        pipeline._segmentation = Inference(
            finetuned,
            duration=pipeline._segmentation.duration,
            step=pipeline._segmentation.step,
            skip_aggregation=pipeline._segmentation.skip_aggregation,
            batch_size=pipeline._segmentation.batch_size,
        )
    return pipeline.to(torch.device(os.getenv("DIARIZATION_DEVICE", "cuda:0")))


def diarize_turns(audio_path: str) -> list[tuple[float, float, str]]:
    """Return ordered, non-empty speaker turns for any mono or stereo recording."""
    import soundfile as sf

    source = Path(audio_path)
    normalized = source.with_name(f"{source.stem}_diarization.wav")
    command = [
        "ffmpeg", "-nostdin", "-fflags", "+discardcorrupt", "-err_detect", "ignore_err",
        "-i", str(source), "-vn", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        "-y", "-loglevel", "error", str(normalized),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Could not decode audio for diarization: {result.stderr.strip()}")
    try:
        samples, sample_rate = sf.read(normalized, always_2d=True)
    finally:
        normalized.unlink(missing_ok=True)
    waveform = torch.from_numpy(samples.T).float()
    output = _pipeline()({"waveform": waveform, "sample_rate": sample_rate})
    annotation = getattr(output, "speaker_diarization", output)
    raw = [(float(turn.start), float(turn.end), str(speaker)) for turn, _, speaker in annotation.itertracks(yield_label=True)]
    identities = {speaker: f"Speaker {index}" for index, speaker in enumerate(dict.fromkeys(s for _, _, s in raw), 1)}
    return [(start, end, identities[speaker]) for start, end, speaker in raw if end > start]


def transcribe_diarized(audio_path: str, transcriber, language: str | None, temp_dir: Path, model_label: str):
    """Diarize first, then transcribe each speaker turn instead of broad time chunks."""
    from .audio import AudioPreprocessor
    from .pipeline import CallTranscript, TranscriptionPipeline

    source = Path(audio_path)
    turns = diarize_turns(str(source))
    if not turns:
        raise RuntimeError("No speaker turns detected")
    duration = AudioPreprocessor.duration(str(source))
    segments: list[dict] = []
    work = Path(temp_dir) / f"diarized_{source.stem}"
    work.mkdir(parents=True, exist_ok=True)
    for index, (start, end, speaker) in enumerate(turns):
        # Preserve brief participants; under 250 ms has no usable lexical content.
        if end - start < 0.25:
            continue
        clip = work / f"turn_{index:04d}.wav"
        command = [
            "ffmpeg", "-nostdin", "-ss", f"{start:.3f}", "-i", str(source),
            "-t", f"{end - start:.3f}", "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            "-y", "-loglevel", "error", str(clip),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg diarized turn failed: {result.stderr}")
        response = transcriber.transcribe(str(clip), language=language, speaker_labels=(speaker, speaker))
        for segment in response.get("segments", []):
            text = (segment.get("text") or "").strip()
            if text:
                segments.append({**segment, "start": round(start + float(segment.get("start", 0)), 2), "end": round(start + float(segment.get("end", end - start)), 2), "speaker": speaker, "text": text})
    full_text = TranscriptionPipeline._labeled_text(segments)
    return CallTranscript(
        call_id=TranscriptionPipeline._call_id(source.name), filename=source.name,
        date=datetime.fromtimestamp(source.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        duration_seconds=round(duration, 2), language_detected=language or "bn", language_confidence=1.0,
        segments=segments, full_text=full_text, word_count=len(full_text.split()),
        processing_time_seconds=0.0, model_used=model_label, status="success", speakers_separated=True,
    )
