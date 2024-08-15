# Build stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

WORKDIR /app

# Install Python and necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=100
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Copy Python and installed packages from builder
COPY --from=builder /usr/bin/python3 /usr/bin/python3
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY app_server.py vector_embed.py ./

# Copy pre-built embeddings
COPY nomic_embed /app/nomic_embed

# Create necessary directories
RUN mkdir -p /app/docs

# Expose port for the app
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "app_server.py"]