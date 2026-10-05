# syntax=docker/dockerfile:1
#
# CPU-only image for the Light-RAG API.
#
#   docker build -t light-rag .
#   docker run --rm -p 8000:8000 \
#       -v "$PWD/data:/app/data" \
#       -v light-rag-models:/models \
#       light-rag
#
# /app/data/docs holds your documents; the FAISS index is written to /app/data/index.
# /models caches downloaded GGUF and embedding weights across container restarts.
# Bake weights into the image for offline/edge use with:  --build-arg BAKE_MODELS=1

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch first; otherwise sentence-transformers pulls the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# llama.cpp is compiled from source. GGML_NATIVE=ON tunes it for the build host's CPU
# (fastest when you build where you run). For an image that runs on other x86 machines,
# pass e.g. --build-arg LLAMA_CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
ARG LLAMA_CMAKE_ARGS="-DGGML_NATIVE=ON"
COPY requirements.txt .
RUN CMAKE_ARGS="${LLAMA_CMAKE_ARGS}" pip install -r requirements.txt

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp: OpenMP runtime used by llama.cpp's CPU backend.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /models /app/data/docs /app/data/index \
    && chown -R app:app /models /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HOST=0.0.0.0 \
    PORT=8000 \
    LIGHTRAG_MODEL=qwen2.5-1.5b \
    LIGHTRAG_QUANT=q4_k_m \
    LIGHTRAG_DOCS_DIR=/app/data/docs \
    LIGHTRAG_INDEX_DIR=/app/data/index

WORKDIR /app
USER app

COPY --chown=app:app src/ src/
COPY --chown=app:app app.py .

# Optionally pre-download the GGUF model and embedding model so the container
# starts without network access. Adds ~1 GB for q4_k_m.
ARG BAKE_MODELS=0
RUN if [ "$BAKE_MODELS" = "1" ]; then \
      python -c "\
import os; \
from huggingface_hub import hf_hub_download; \
from sentence_transformers import SentenceTransformer; \
from src.core.quant_engine import MODEL_PRESETS; \
from src.retrieval.vector_store import DEFAULT_MODEL_NAME; \
spec = MODEL_PRESETS[os.environ['LIGHTRAG_MODEL']]; \
hf_hub_download(spec.repo_id, spec.files[os.environ['LIGHTRAG_QUANT']]); \
SentenceTransformer(DEFAULT_MODEL_NAME, device='cpu')"; \
    fi

VOLUME ["/models", "/app/data"]
EXPOSE 8000

# Generous start period: the first start downloads the model and builds the index.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["python", "app.py"]
