# Docker deployment

This repo runs the `whisper-bn` FastAPI transcription service in Docker.

## Build and run

```bash
docker compose up -d --build call-intelligence-pipeline
```

Health check:

```bash
curl http://SERVER_IP:3433/health
```

## OpenAI-compatible transcription

Use Bearer auth for every `/v1/*` request. Routers that require model discovery can call `GET /v1/models`; it returns `whisper-bn` and `whisper-1`.

```bash
curl http://SERVER_IP:3433/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_SITE_API_KEY" \
  -F "file=@samples/call.wav" \
  -F "model=whisper-bn" \
  -F "language=bn"
```

For long audio, use the same endpoint asynchronously:

```bash
curl http://SERVER_IP:3433/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_SITE_API_KEY" \
  -F "file=@samples/long-call.wav" \
  -F "async=true" \
  -F "diarize=true" \
  -F "labels=Agent,Customer"
```

Poll or download async results:

```bash
curl -H "Authorization: Bearer YOUR_SITE_API_KEY" http://SERVER_IP:3433/v1/transcriptions/JOB_ID
curl -H "Authorization: Bearer YOUR_SITE_API_KEY" http://SERVER_IP:3433/v1/transcriptions/JOB_ID/result
curl -H "Authorization: Bearer YOUR_SITE_API_KEY" http://SERVER_IP:3433/v1/transcriptions/JOB_ID/text
```

## Volumes

- `./samples` mounts read-only to `/app/samples` for optional local input audio.
- `./transcripts` mounts read-write to `/app/transcripts` for uploads, SQLite queue state, and outputs.
- `call-intelligence-models` mounts at `/models` for Hugging Face cache.

Do not prune `call-intelligence-models` if you want to preserve model downloads.

## Settings

```bash
PORT=3433
SITE_API_KEY=change-me
MODEL_PROVIDER=whisper-bn
WHISPER_MODEL=bitwisemind/sam_15000_clean_text_full_model
ASR_QUEUE_DB=/app/transcripts/transcription_queue.sqlite3
ASR_UPLOAD_DIR=/app/transcripts/uploads
ASR_OUTPUT_DIR=/app/transcripts
CUDA_VISIBLE_DEVICES=0
```

## Host reverse proxy

This compose file uses `network_mode: host` because the current server's Docker bridge DNS cannot resolve package repositories or external API hosts. The app listens on `0.0.0.0:${PORT:-3433}` on the host.
