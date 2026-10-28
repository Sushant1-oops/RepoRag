# Multi-stage build: node compiles the frontend, python runs the backend.
# The final image serves both the API and the built frontend on :8000, so the
# deployed app is a single origin (no CORS and no mixed-content concerns).

# ---- stage 1: frontend -------------------------------------------------------
FROM node:24-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# API_BASE stays empty: requests are same-origin, served by FastAPI below.
RUN npm run build

# ---- stage 2: backend --------------------------------------------------------
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # keep every HF download in one explicit place, preheated at build time
    HF_HOME=/opt/hf

# requirements.txt does not pin torch, and on Linux the default wheel set
# pulls multi-GB CUDA libraries; the CPU-only wheel is all this app needs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY repo_rag/ ./repo_rag/
COPY server.py ./
COPY --from=frontend /fe/dist ./frontend/dist

# Preheat the embedding + reranker models into the image layer so a freshly
# started instance answers immediately instead of downloading ~180MB first.
RUN python -c "from repo_rag import ingest, retrieval; ingest.get_embedder(); retrieval._reranker()"

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
