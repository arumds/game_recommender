# Serving image only: no training dependencies (no torch, implicit, ranx).
# Train first (python src/main.py) so models/ exists, then mount it at run time:
#   docker build -t steam-reco-api .
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models" steam-reco-api
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src/ src/
ENV PYTHONUNBUFFERED=1 MODELS_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
