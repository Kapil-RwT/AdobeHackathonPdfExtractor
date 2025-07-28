FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential pkg-config libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py round_1a_extractor.py round_1b_extractor.py ./
COPY minilm-sentence-transformer.onnx ./
COPY models/minilm ./models/minilm

ENTRYPOINT ["python", "main.py"]
