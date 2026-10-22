# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Prepare data and verify trained models
RUN python scripts/prepare_data.py --subset FD001 && \
    python scripts/train_model.py --subset FD001

EXPOSE 8000 8501

# Default command launches FastAPI server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
