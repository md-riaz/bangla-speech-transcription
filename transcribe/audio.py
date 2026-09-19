"""
Audio preprocessing and stereo channel handling (ffmpeg-based).

Two things matter for accuracy on real call recordings:

1. Resampling to 16 kHz mono. Whisper's VAD expects 16 kHz; feeding it raw
   8 kHz phone audio can make it return zero segments (looks like silence).

2. Loudness normalization. Phone recordings are often very quiet (mean levels
   around -40 dB). A fixed volume multiplier is unreliable; ffmpeg's `loudnorm`
   lifts any source to a consistent target level.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Telephony band-pass (200 Hz - 3400 Hz) + EBU R128 loudness normalization.
# Used for every conversion so noisy/quiet calls are cleaned consistently.
DEFAULT_FILTER = "highpass=f=200,lowpass=f=3400,loudnorm=I=-16:TP=-1.5:LRA=11"


class AudioPreprocessor:
    SUPPORTED = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac", ".gsm", ".amr", ".webm"}

    @staticmethod
    def convert(input_path, output_dir, audio_filter: str = DEFAULT_FILTER) -> str:
        """Convert any input to a 16 kHz mono WAV, cleaned and normalized."""
        p = Path(input_path)
        out = Path(output_dir) / f"{p.stem}_16k.wav"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        cmd = [
            "ffmpeg", "-i", str(p),
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            "-af", audio_filter,
            str(out), "-y", "-loglevel", "error",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {r.stderr}")
        return str(out)

    @staticmethod
    def duration(path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except ValueError:
            return 0.0

    @staticmethod
    def channels(path) -> int:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return int(r.stdout.strip().split("\n")[0])
        except (ValueError, IndexError):
            return 1

    @classmethod
    def supported(cls, path) -> bool:
        p = Path(path)
        if p.suffix.lower() in cls.SUPPORTED:
            return True
        if not p.is_file():
            return False

        # Some downloaded recordings arrive without an extension even though
        # the container format is supported (for example, MP3 saved as
        # ``savedly_sample``). Probe the media content before rejecting it so
        # CLI single-file processing follows the same reality as ffmpeg.
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1", str(p),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and "audio" in r.stdout.lower().split()


class StereoSplitter:
    """Split a stereo recording into two mono channels.

    Many call-center systems record the two parties on separate channels
    (e.g. left = agent, right = customer). Transcribing each channel on its own
    gives far cleaner per-speaker text than mixing them down to mono.
    """

    @staticmethod
    def is_stereo(path) -> bool:
        return AudioPreprocessor.channels(path) == 2

    @staticmethod
    def split(path, tmpdir, audio_filter: str = DEFAULT_FILTER) -> tuple[str, str]:
        """Return (left_wav, right_wav), each 16 kHz mono and normalized."""
        stem = Path(path).stem
        left = Path(tmpdir) / f"{stem}_left.wav"
        right = Path(tmpdir) / f"{stem}_right.wav"
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        for out, ch in [(left, "c0=c0"), (right, "c0=c1")]:
            out.unlink(missing_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-i", str(path),
                    # Per-channel band-pass + loudness normalization. Split
                    # channels are often quiet on their own, so normalizing each
                    # independently matters.
                    "-af", f"pan=mono|{ch},{audio_filter}",
                    "-ar", "16000", "-acodec", "pcm_s16le",
                    str(out), "-y", "-loglevel", "error",
                ],
                check=True,
            )
        return str(left), str(right)

    @staticmethod
    def merge(left_segs, right_segs, labels=("Speaker A", "Speaker B")) -> list:
        """Interleave two channels' segments into one timeline, labelled."""
        a, b = labels
        labeled = (
            [{**s, "speaker": a} for s in left_segs]
            + [{**s, "speaker": b} for s in right_segs]
        )
        return sorted(labeled, key=lambda x: x["start"])
