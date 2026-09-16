# Bangla Speech Transcription

OpenAI-compatible Bengali audio transcription service backed by the local BitwiseMind SAM15K Whisper model, with optional Pyannote speaker separation and optional Gemini transcription/summary mode for the web UI.

## API

Authentication is strict OpenAI-style Bearer auth:

```http
Authorization: Bearer YOUR_SITE_API_KEY
```

Canonical transcription endpoint:

```http
POST /v1/audio/transcriptions
```

Multipart fields:

| Field | Default | Notes |
| --- | --- | --- |
| `file` | required | Audio file: wav, mp3, m4a, flac, ogg, webm, etc. |
| `model` | `whisper-bn` | Accepts `whisper-bn`, `whisper-1`, or a Gemini model when `engine=gemini`. |
| `language` | `bn` | Use `bn` or `auto`. |
| `diarize` | `false` | `true` enables local Pyannote speaker separation for local SAM15K. |
| `labels` | `Agent,Customer` | Speaker labels for diarized output. |
| `async` | `false` | `true` returns a job id for long audio. |
| `engine` | `whisper-bn` | `whisper-bn` for local SAM15K, `gemini` for Gemini transcription. |
| `gemini_api_key` | empty | Required only for BYOK Gemini transcription when no server Gemini key is configured. |

### Synchronous transcription

Best for short voice notes and integrations that expect an immediate OpenAI-style response:

```bash
curl https://bntranscription.ai-api.ancbd.com/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_SITE_API_KEY" \
  -F "file=@voice.mp3" \
  -F "model=whisper-bn" \
  -F "language=bn"
```

Response:

```json
{
  "text": "পূর্ণ ট্রান্সক্রিপ্ট...",
  "language": "bn",
  "duration": 38.2,
  "model": "whisper-bn/bitwisemind/sam_15000_clean_text_full_model",
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.2, "text": "আসসালামু আলাইকুম", "speaker": "Speaker 1"}
  ]
}
```

### Async transcription for long audio

Use the same endpoint with `async=true` for long calls/meetings:

```bash
curl https://bntranscription.ai-api.ancbd.com/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_SITE_API_KEY" \
  -F "file=@long-call.mp3" \
  -F "async=true" \
  -F "diarize=true" \
  -F "labels=Agent,Customer"
```

Response:

```json
{"job_id":"...","status":"queued","engine":"whisper-bn"}
```

Poll the job:

```bash
curl -H "Authorization: Bearer YOUR_SITE_API_KEY" \
  https://bntranscription.ai-api.ancbd.com/v1/transcriptions/JOB_ID
```

Download artifacts:

```text
GET /v1/transcriptions/{job_id}/result
GET /v1/transcriptions/{job_id}/text
GET /v1/transcriptions/{job_id}/srt
```

The web UI uses the same OpenAI-compatible endpoint with `async=true` and `diarize=true`.

## Local development

```bash
pip install ".[sam15000,api,diarization,gemini]"
uvicorn transcribe.api:app --host 0.0.0.0 --port 3433
```

Docker:

```bash
docker compose up -d --build call-intelligence-pipeline
```

## CLI

```bash
transcribe --file call.wav --language bn --labels "Agent,Customer" --output transcripts
transcribe --input samples --language bn --labels "Agent,Customer" --output transcripts
```

## Output files

For each successful transcription, the pipeline writes:

- `<call_id>.json` structured transcript
- `<call_id>.txt` readable transcript
- `<call_id>.srt` subtitles

Audio samples, transcript outputs, model caches, and secrets are git-ignored and must not be committed.
