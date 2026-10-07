FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DATA_DIR=/data/indexes

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only torch first, so sentence-transformers does not pull the multi-GB CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

RUN mkdir -p /data/indexes /home/app/.cache/huggingface && chown -R app:app /data /home/app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
