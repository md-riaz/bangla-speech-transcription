"""
Transcription pipeline: orchestrates preprocessing, transcription, and output
writing (JSON + TXT + SRT).

Transcription backends are pluggable. The public service defaults to
whisper-bn, a local GPU Bengali Whisper ASR backend.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .audio import AudioPreprocessor

log = logging.getLogger(__name__)


@dataclass
class CallTranscript:
    call_id: str
    filename: str
    date: str
    duration_seconds: float
    language_detected: str
    language_confidence: float
    segments: list
    full_text: str
    word_count: int
    processing_time_seconds: float
    model_used: str
    status: str
    speakers_separated: bool = False
    error: Optional[str] = None


def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class TranscriptionPipeline:
    def __init__(
        self,
        output_dir: str = "./transcripts",
        temp_dir: Optional[str] = None,
        language: Optional[str] = None,
        speaker_labels: tuple = ("Speaker A", "Speaker B"),
        separate_speakers: bool = True,
        write_srt: bool = True,
        google_api_key: Optional[str] = None,
        gemini_model_id: str = "gemini-3.1-flash-lite",
        engine: str = "whisper-bn",
        whisper_model_id: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir or output_dir) / "_temp"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.language = language
        self.speaker_labels = speaker_labels
        self.separate_speakers = separate_speakers
        self.write_srt = write_srt

        from .backends import create_backend

        self.engine = engine
        self.transcriber = create_backend(
            engine,
            google_api_key=google_api_key,
            gemini_model_id=gemini_model_id,
            whisper_model_id=whisper_model_id,
            temp_dir=str(self.temp_dir),
        )
        self._model_label = f"{self.transcriber.name}/{self.transcriber.model_id}"

        self.processed_log = self.output_dir / "processed_files.json"
        self.processed = self._load_processed()

    # --- bookkeeping ---------------------------------------------------------
    def _load_processed(self) -> set:
        if not self.processed_log.exists():
            return set()
        try:
            with open(self.processed_log) as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            log.warning("Ignoring corrupt processed log: %s", self.processed_log)
            return set()

    def _save_processed(self) -> None:
        tmp = self.processed_log.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(list(self.processed), f)
        os.replace(tmp, self.processed_log)

    @staticmethod
    def _call_id(filename) -> str:
        return (
            Path(filename).stem
            .replace("_16k", "").replace("_left", "").replace("_right", "")
        )

    # --- single file ---------------------------------------------------------
    def process_file(self, audio_path, skip_done: bool = True) -> CallTranscript:
        audio_path = Path(audio_path)
        call_id = self._call_id(audio_path.name)

        if skip_done and call_id in self.processed:
            log.info("Skipping (already done): %s", audio_path.name)
            return self._stub(call_id, audio_path.name, "skipped")

        if not AudioPreprocessor.supported(str(audio_path)):
            return self._stub(
                call_id, audio_path.name, "failed",
                error=f"Unsupported format: {audio_path.suffix}",
            )

        t0 = time.time()
        try:
            file_date = datetime.fromtimestamp(
                os.path.getmtime(str(audio_path))
            ).strftime("%Y-%m-%d %H:%M:%S")

            r = self.transcriber.transcribe(
                str(audio_path),
                language=self.language,
                speaker_labels=self.speaker_labels,
            )
            segs = r["segments"]
            full_text = r["full_text"]
            duration = r["duration_seconds"]
            lang, lang_conf = r["language_detected"], r["language_confidence"]

            transcript = CallTranscript(
                call_id=call_id, filename=audio_path.name, date=file_date,
                duration_seconds=duration, language_detected=lang,
                language_confidence=lang_conf, segments=segs, full_text=full_text,
                word_count=len(full_text.split()),
                processing_time_seconds=round(time.time() - t0, 2),
                model_used=self._model_label, status="success",
                speakers_separated=r["speakers_separated"],
            )
            self._save_outputs(transcript)
            self.processed.add(call_id)
            self._save_processed()
            log.info(
                "  Done: %.0fs audio, %d words, lang=%s (%.0f%%), took %.1fs",
                duration, transcript.word_count, lang, lang_conf * 100,
                transcript.processing_time_seconds,
            )
            return transcript

        except Exception as e:  # noqa: BLE001 - we want to record any failure
            log.error("  Failed: %s — %s", audio_path.name, e)
            return self._stub(
                call_id, audio_path.name, "failed", error=str(e),
                proc_time=round(time.time() - t0, 2),
            )

    def _stub(self, call_id, filename, status, error=None, proc_time=0.0):
        return CallTranscript(
            call_id=call_id, filename=filename, date="", duration_seconds=0,
            language_detected="", language_confidence=0, segments=[], full_text="",
            word_count=0, processing_time_seconds=proc_time,
            model_used=self._model_label, status=status, error=error,
        )

    # --- output --------------------------------------------------------------
    @staticmethod
    def _labeled_text(segs) -> str:
        lines, cur_spk, cur_parts = [], None, []
        for seg in segs:
            spk, txt = seg.get("speaker", ""), seg.get("text", "").strip()
            if not txt:
                continue
            if spk != cur_spk:
                if cur_parts:
                    lines.append(f"[{cur_spk}]: {' '.join(cur_parts)}")
                cur_spk, cur_parts = spk, [txt]
            else:
                cur_parts.append(txt)
        if cur_parts:
            lines.append(f"[{cur_spk}]: {' '.join(cur_parts)}")
        return "\n".join(lines)

    def _save_outputs(self, t: CallTranscript) -> None:
        base = self.output_dir / t.call_id

        with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(asdict(t), f, ensure_ascii=False, indent=2)

        with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Call ID  : {t.call_id}\n")
            f.write(f"File     : {t.filename}\n")
            f.write(f"Date     : {t.date}\n")
            f.write(
                f"Duration : {t.duration_seconds:.0f}s "
                f"({t.duration_seconds / 60:.1f} min)\n"
            )
            f.write(f"Language : {t.language_detected} ({t.language_confidence:.0%})\n")
            f.write(f"Words    : {t.word_count}\n")
            f.write(f"Model    : {t.model_used}\n")
            f.write(f"{'=' * 60}\n\nTRANSCRIPT:\n\n{t.full_text}\n\n")
            f.write(f"\n{'-' * 60}\nTIMED SEGMENTS:\n\n")
            for seg in t.segments:
                m1, s1 = divmod(int(seg["start"]), 60)
                m2, s2 = divmod(int(seg["end"]), 60)
                spk = seg.get("speaker", "")
                tag = f" [{spk}]" if spk else ""
                f.write(f"[{m1:02d}:{s1:02d}->{m2:02d}:{s2:02d}]{tag}\n{seg['text']}\n\n")

        if self.write_srt:
            with open(base.with_suffix(".srt"), "w", encoding="utf-8") as f:
                for i, seg in enumerate(t.segments, 1):
                    spk = seg.get("speaker", "")
                    prefix = f"{spk}: " if spk else ""
                    f.write(
                        f"{i}\n"
                        f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}\n"
                        f"{prefix}{seg['text']}\n\n"
                    )

        log.info("  Saved: %s.json + .txt%s", base.stem, " + .srt" if self.write_srt else "")

    # --- batch ---------------------------------------------------------------
    def process_directory(self, input_dir, days: Optional[int] = None) -> dict:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        all_files = []
        for ext in AudioPreprocessor.SUPPORTED:
            all_files += list(input_dir.rglob(f"*{ext}"))
            all_files += list(input_dir.rglob(f"*{ext.upper()}"))

        if days:
            cutoff = time.time() - days * 86400
            all_files = [f for f in all_files if os.path.getmtime(str(f)) >= cutoff]

        files = sorted(set(all_files), key=lambda f: os.path.getmtime(str(f)))
        if not files:
            log.warning("No supported audio files found%s", f" in last {days} days" if days else "")
            return {"success": 0, "failed": 0, "skipped": 0}

        total_dur = sum(AudioPreprocessor.duration(str(f)) for f in files)
        log.info(
            "Found %d files | %.0f min audio", len(files), total_dur / 60
        )

        results = {"success": 0, "failed": 0, "skipped": 0}
        for i, f in enumerate(files, 1):
            log.info("[%d/%d] %s", i, len(files), f.name)
            t = self.process_file(str(f))
            results[t.status] = results.get(t.status, 0) + 1

        log.info(
            "\nDone — Success:%d Failed:%d Skipped:%d",
            results["success"], results["failed"], results["skipped"],
        )
        return results
