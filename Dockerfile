# Stage 1: export dependencies from poetry
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir poetry==2.1.3 poetry-plugin-export

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


# Stage 2: runtime image
FROM python:3.12-slim

# libgomp1 required by torch (OpenMP), libglib2.0-0 by some ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/requirements.txt ./

# Install CPU-only torch to keep image size manageable on a VPS without GPU
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY src/ ./src/
COPY config.toml ./

ENV HF_HOME=/app/.cache/huggingface

RUN useradd --no-create-home --shell /bin/false appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.bot"]