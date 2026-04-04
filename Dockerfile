FROM python:3.11-slim

# Set metadata
LABEL maintainer="shaikhakramshakil"
LABEL description="OpenEnv Data Cleaning Environment"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces uses port 7860 by default
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100
ENV OPENENV_TASK=task_1_identify

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 2
