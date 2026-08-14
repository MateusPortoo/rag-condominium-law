FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY scripts/ scripts/

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "rag_condominios.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
