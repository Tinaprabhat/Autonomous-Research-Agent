# Single container: FastAPI serves the API + static frontend.
# Targets Hugging Face Spaces (Docker SDK, CPU Basic): port 7860, non-root user.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

USER user
WORKDIR /home/user/app

# CPU-only torch first (much smaller than the default CUDA wheel), then the rest.
COPY --chown=user requirements.txt .
RUN pip install --user torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements.txt

COPY --chown=user backend ./backend
COPY --chown=user frontend ./frontend
COPY --chown=user scripts/download_models.py ./scripts/download_models.py
RUN python scripts/download_models.py
# Models are baked into the image: never touch the network for them at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Secrets (GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY) are injected as env vars at runtime.
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:'+__import__('os').environ.get('PORT','7860')+'/health').status==200 else 1)"
# Render injects $PORT; falls back to 7860 for Hugging Face Spaces.
CMD ["sh", "-c", "uvicorn backend.app.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
