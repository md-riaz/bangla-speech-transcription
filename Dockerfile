# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip python-is-python3 ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel
COPY pyproject.toml README.md LICENSE requirements.txt ./
COPY transcribe ./transcribe
COPY demo/requirements.txt ./demo/requirements.txt
RUN /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio==2.11.0+cu128 \
 && /opt/venv/bin/pip install '.[sam15000,api,diarization,gemini]' \
 && /opt/venv/bin/pip install -r requirements.txt \
 && /opt/venv/bin/pip install -r demo/requirements.txt \
 && /opt/venv/bin/pip uninstall -y torchcodec
ENV PATH=/opt/venv/bin:$PATH PYANNOTE_MODEL_DIR=/models/pyannote/community-1 DIARIZATION_DEVICE=cuda:0
CMD ["sh", "-c", "uvicorn transcribe.api:app --host 0.0.0.0 --port ${PORT:-3433}"]
