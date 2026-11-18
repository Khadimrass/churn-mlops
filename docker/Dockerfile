# syntax=docker/dockerfile:1
# Image multi-cibles :
#   --target serve : API légère (FastAPI + modèle), utilisateur non-root
#   --target train : pipeline d'entraînement + serveur MLflow

ARG PYTHON_VERSION=3.12

# ---------- base commune ----------
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHURN_HOME=/app \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_AGENT_HINT=1
# libgomp1 : requis par LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------- API ----------
FROM base AS serve
COPY requirements/serve.txt requirements/serve.txt
RUN pip install -r requirements/serve.txt
COPY configs ./configs
COPY src ./src
# Le modèle champion est embarqué : chaque image = une version de modèle immuable
COPY models ./models
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data/production reports \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Entraînement / MLflow ----------
FROM base AS train
COPY requirements ./requirements
RUN pip install -r requirements/train.txt
COPY configs ./configs
COPY src ./src
CMD ["python", "-m", "churn.train"]
