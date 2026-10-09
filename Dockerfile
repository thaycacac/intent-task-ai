FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY openapi ./openapi
COPY data/mappings ./data/mappings
COPY artifacts ./artifacts

ENV PYTHONPATH=/app/src
ENV ARTIFACTS_DIR=/app/artifacts/category/latest
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "intent_task_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
