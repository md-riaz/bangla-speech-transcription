"""Lightweight smoke tests — no API calls or audio required.

These verify the package imports, the CLI parses arguments, and the pure
helpers behave. Run with: pytest
"""

import json

from transcribe import __version__
from transcribe.audio import AudioPreprocessor, StereoSplitter
from transcribe.cli import build_parser
from transcribe.analyze_cli import build_parser as build_analyze_parser
from transcribe.pipeline import _srt_timestamp


def test_version():
    assert isinstance(__version__, str) and __version__


def test_supported_formats():
    assert AudioPreprocessor.supported("a.wav")
    assert AudioPreprocessor.supported("CALL.MP3")
    assert not AudioPreprocessor.supported("notes.txt")


def test_convert_rebuilds_stale_temp_file(tmp_path, monkeypatch):
    source = tmp_path / "clip.wav"
    source.write_bytes(b"new audio")
    stale = tmp_path / "clip_16k.wav"
    stale.write_bytes(b"old audio")

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, capture_output, text):
        stale.write_bytes(b"rebuilt audio")
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)
    assert AudioPreprocessor.convert(str(source), tmp_path) == str(stale)
    assert stale.read_bytes() == b"rebuilt audio"


def test_supported_extensionless_audio_probe(tmp_path, monkeypatch):
    audio = tmp_path / "savedly_sample"
    audio.write_bytes(b"not actually decoded because ffprobe is mocked")

    class Result:
        returncode = 0
        stdout = "audio\n"

    def fake_run(cmd, capture_output, text):
        assert cmd[0] == "ffprobe"
        assert str(audio) == cmd[-1]
        assert capture_output is True
        assert text is True
        return Result()

    monkeypatch.setattr("transcribe.audio.subprocess.run", fake_run)

    assert AudioPreprocessor.supported(audio)


def test_unsupported_extensionless_file_probe_failure(tmp_path, monkeypatch):
    unknown = tmp_path / "notes"
    unknown.write_text("plain text")

    class Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr("transcribe.audio.subprocess.run", lambda *a, **k: Result())

    assert not AudioPreprocessor.supported(unknown)


def test_srt_timestamp():
    assert _srt_timestamp(0) == "00:00:00,000"
    assert _srt_timestamp(3661.5) == "01:01:01,500"


def test_merge_orders_by_start_and_labels():
    left = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    right = [{"start": 0.5, "end": 1.5, "text": "yo"}]
    merged = StereoSplitter.merge(left, right, labels=("Agent", "Customer"))
    assert [s["speaker"] for s in merged] == ["Agent", "Customer"]
    assert merged[0]["start"] <= merged[1]["start"]


def test_cli_parses_file_arg():
    args = build_parser().parse_args(["--file", "x.wav", "--language", "bn"])
    assert args.file == "x.wav"
    assert args.language == "bn"


def test_cli_parses_google_api_key():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--language", "bn", "--google-api-key", "k"]
    )
    assert args.google_api_key == "k"


def test_cli_model_flag():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--model", "gemini-3.1-flash-lite"]
    )
    assert args.model == "gemini-3.1-flash-lite"


def test_cli_model_flag_25flash():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--model", "gemini-2.5-flash"]
    )
    assert args.model == "gemini-2.5-flash"


def test_cli_requires_a_source():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_analyze_cli_parses_file_arg():
    args = build_analyze_parser().parse_args(["--file", "transcripts/call.json"])
    assert args.file == "transcripts/call.json"
    assert not args.reanalyze


def test_analyze_cli_parses_batch_args():
    args = build_analyze_parser().parse_args(
        ["--input", "transcripts/", "--reanalyze", "--no-csv"]
    )
    assert args.input == "transcripts/"
    assert args.reanalyze
    assert args.no_csv


def test_analyze_cli_openai_compatible_flags():
    args = build_analyze_parser().parse_args(
        [
            "--file", "t.json",
            "--api-key", "sk-test",
            "--base-url", "https://llm.example.com/v1",
            "--model", "my-model",
        ]
    )
    assert args.api_key == "sk-test"
    assert args.base_url == "https://llm.example.com/v1"
    assert args.model == "my-model"


def test_call_analyzer_requires_api_key(monkeypatch):
    import pytest
    from transcribe.analyze import CallAnalyzer

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        CallAnalyzer()


def test_call_analyzer_extracts_openai_compatible_content(monkeypatch):
    from transcribe.analyze import CallAnalyzer

    analyzer = CallAnalyzer(api_key="sk-test", model_id="model-x", base_url="https://llm.example.com/v1")

    class Resp:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": '{"agent_score":91}'}}]
            }).encode("utf-8")

    def fake_urlopen(req, timeout):
        assert req.full_url == "https://llm.example.com/v1/chat/completions"
        assert timeout == 120
        return Resp()

    monkeypatch.setattr("transcribe.analyze.urllib.request.urlopen", fake_urlopen)
    assert analyzer._call_model("prompt") == '{"agent_score":91}'


def test_analyze_csv_columns():
    from transcribe.analyze import _CSV_COLUMNS
    required = {"call_id", "brand", "issue_category", "resolution",
                "agent_score", "customer_sentiment", "coaching_tip"}
    assert required.issubset(set(_CSV_COLUMNS))


def test_key_pool_single_key():
    from transcribe.key_pool import GeminiKeyPool
    pool = GeminiKeyPool(["fake-key-1"])
    assert pool.total_count == 1
    assert pool.active_count == 1


def test_key_pool_deduplicates():
    from transcribe.key_pool import GeminiKeyPool
    pool = GeminiKeyPool(["k1", "k2", "k1", "k3"])
    assert pool.total_count == 3


def test_key_pool_rotation():
    from transcribe.key_pool import GeminiKeyPool, AllKeysExhaustedError
    pool = GeminiKeyPool(["k1", "k2"])
    pool._clients = {0: "client0", 1: "client1"}
    _, idx = pool.get_client()
    assert idx == 0
    pool.mark_exhausted(0)
    _, idx = pool.get_client()
    assert idx == 1
    pool.mark_exhausted(1)
    try:
        pool.get_client()
        assert False, "Should have raised AllKeysExhaustedError"
    except AllKeysExhaustedError:
        pass


def test_cli_parses_engine_whisper_sam15000():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "whisper-sam15000", "--language", "bn"]
    )
    assert args.engine == "whisper-sam15000"
    assert args.language == "bn"




def test_cli_parses_engine_whisper_tugstugi():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "whisper-tugstugi", "--language", "bn"]
    )
    assert args.engine == "whisper-tugstugi"
    assert args.language == "bn"


def test_tugstugi_factory_uses_default_model(monkeypatch):
    from transcribe.backends import base

    captured = {}

    class FakeBackend:
        def __init__(self, model_id=None, temp_dir=None, provider_name=""):
            captured["model_id"] = model_id
            captured["provider_name"] = provider_name

    monkeypatch.setattr(
        "transcribe.backends.whisper_sam15000.WhisperSam15000Backend", FakeBackend
    )
    backend = base.create_backend("whisper-tugstugi")
    assert isinstance(backend, FakeBackend)
    assert captured["model_id"] == "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"
    assert captured["provider_name"] == "whisper-tugstugi"


def test_cli_parses_engine_whisper_bn():
    args = build_parser().parse_args(
        ["--file", "x.wav", "--engine", "whisper-bn", "--language", "bn"]
    )
    assert args.engine == "whisper-bn"
    assert args.language == "bn"


def test_backend_factory_whisper_bn_alias(monkeypatch):
    from transcribe.backends import base

    captured = {}

    class FakeBackend:
        def __init__(self, model_id=None, temp_dir=None, provider_name=""):
            captured["model_id"] = model_id
            captured["provider_name"] = provider_name

    monkeypatch.setattr(
        "transcribe.backends.whisper_sam15000.WhisperSam15000Backend", FakeBackend
    )
    backend = base.create_backend("whisper-bn")
    assert isinstance(backend, FakeBackend)
    assert captured["provider_name"] == "whisper-bn"


def test_queue_lifecycle(tmp_path):
    from transcribe.queue import TranscriptionQueue

    queue = TranscriptionQueue(str(tmp_path / "queue.sqlite3"))
    job = queue.create_job("audio.wav", str(tmp_path), language="bn")
    assert queue.get_job(job.id).status == "queued"
    claimed = queue.claim_next()
    assert claimed.id == job.id
    assert claimed.status == "processing"
    assert queue.requeue_processing_jobs() == 1
    assert queue.get_job(job.id).status == "queued"
    claimed = queue.claim_next()
    assert claimed.id == job.id
    assert claimed.status == "processing"
    queue.complete_job(job.id, str(tmp_path / "out.json"))
    assert queue.get_job(job.id).status == "completed"


def test_backend_factory_rejects_unknown_engine():
    import pytest
    from transcribe.backends.base import create_backend

    with pytest.raises(ValueError):
        create_backend("not-real")


def test_whisper_timestamp_pair():
    from transcribe.backends.whisper_sam15000 import _timestamp_pair

    assert _timestamp_pair((None, 2.5)) == (0.0, 2.5)
    assert _timestamp_pair((1, None)) == (1.0, 1.0)


def test_whisper_segments_to_text():
    from transcribe.backends.whisper_sam15000 import _segments_to_text

    text = _segments_to_text([{"speaker": "Agent", "text": "হ্যালো"}])
    assert text == "[Agent]: হ্যালো"


def test_queue_completed_job_includes_inline_transcript(tmp_path):
    from transcribe.queue import TranscriptionQueue

    transcript_path = tmp_path / "out.json"
    transcript_path.write_text(
        json.dumps({"call_id": "out", "status": "success", "full_text": "hello"}),
        encoding="utf-8",
    )
    queue = TranscriptionQueue(str(tmp_path / "queue.sqlite3"))
    job = queue.create_job("audio.wav", str(tmp_path), language="bn")
    queue.complete_job(job.id, str(transcript_path))

    data = TranscriptionQueue.job_to_dict(queue.get_job(job.id))

    assert data["engine"] == "whisper-bn"
    assert data["result_json_path"] == str(transcript_path)
    assert data["result_path"] == str(transcript_path)
    assert data["transcript"]["call_id"] == "out"
    assert data["result"] == data["transcript"]



def _assert_no_public_path_or_url_leaks(value):
    forbidden_key_parts = ("path", "dir", "url", "urls")
    forbidden_value_parts = ("/home/", "/app/", "/tmp/", "\\", "http://", "https://")

    def walk(item):
        if isinstance(item, dict):
            for key, nested in item.items():
                assert not any(part in key.lower() for part in forbidden_key_parts), key
                walk(nested)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)
        elif isinstance(item, str):
            assert not any(part in item for part in forbidden_value_parts), item

    walk(value)


def test_api_completed_job_exposes_only_inline_result_and_downloads(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from transcribe import api

    transcript_path = tmp_path / "call.json"
    transcript_path.write_text(
        json.dumps({"call_id": "call", "status": "success", "full_text": "[Agent]: হ্যালো"}),
        encoding="utf-8",
    )
    transcript_path.with_suffix(".txt").write_text("[Agent]: হ্যালো\n", encoding="utf-8")
    transcript_path.with_suffix(".srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nAgent: হ্যালো\n\n",
        encoding="utf-8",
    )

    queue = api.TranscriptionQueue(str(tmp_path / "queue.sqlite3"))
    job = queue.create_job("audio.wav", str(tmp_path), language="bn")
    queue.complete_job(job.id, str(transcript_path))
    monkeypatch.setattr(api, "queue", queue)

    client = TestClient(api.app)
    response = client.get(f"/v1/transcriptions/{job.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"]["full_text"] == "[Agent]: হ্যালো"
    assert data["result"] == data["transcript"]
    _assert_no_public_path_or_url_leaks(data)

    list_response = client.get("/v1/transcriptions")
    assert list_response.status_code == 200
    _assert_no_public_path_or_url_leaks(list_response.json())

    result_response = client.get(f"/v1/transcriptions/{job.id}/result")
    assert result_response.status_code == 200
    assert result_response.json()["call_id"] == "call"

    alias_response = client.get(f"/transcriptions/{job.id}/result")
    assert alias_response.status_code == 404

    text_response = client.get(f"/v1/transcriptions/{job.id}/text")
    assert text_response.status_code == 200
    assert "হ্যালো" in text_response.text

    srt_response = client.get(f"/v1/transcriptions/{job.id}/srt")
    assert srt_response.status_code == 200
    assert "00:00:00,000" in srt_response.text
