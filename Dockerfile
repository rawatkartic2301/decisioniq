# Multi-stage Dockerfile for DecisionIQ API and Streamlit UI
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

# Copy dependency specifications
COPY pyproject.toml .

# Install dependencies into virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -e .

# -------------------------------------------------------------
# Runtime Stage
# -------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source code and data assets
COPY src/ /app/src/
COPY data/ /app/data/
COPY prompts/ /app/prompts/
COPY pyproject.toml /app/

# Install the package itself in editable/develop mode inside venv
RUN pip install --no-deps -e .

EXPOSE 8000 8501

# Default entrypoint starts FastAPI
CMD ["uvicorn", "decisioniq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
