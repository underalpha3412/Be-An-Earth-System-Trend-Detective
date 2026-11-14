# Multi-stage slim Dockerfile for model_go headless meteorological engine
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.162.6 \
    uvicorn==0.54.0 \
    pydantic==2.14.3 \
    numpy==2.5.4 \
    scipy==1.18.1 \
    requests==2.34.2 \
    joblib==1.6.0

# Final runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Copy installed python wheels from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model_go package and model weights
COPY model_go /app/model_go
COPY outputs /app/outputs
COPY data /app/data
COPY test_model_alpha /app/test_model_alpha

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_GO_PROFILE=server \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/v1/health')" || exit 1

ENTRYPOINT ["uvicorn", "model_go.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
