# Start with the NVIDIA CUDA base image that has the necessary drivers
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Set the working directory
WORKDIR /app

# Install Python and necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up-to-date
RUN python3 -m pip install --upgrade pip

# Copy requirements file and install Python dependencies
COPY requirements.txt .

ENV PIP_DEFAULT_TIMEOUT=100

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app_server.py vector_embed.py update_vector_db.sh ./

# Copy pre-built embeddings into the container
COPY nomic_embed /app/nomic_embed

RUN chmod +x update_vector_db.sh

# Create necessary directories
RUN mkdir -p /app/docs

# Expose port for the app
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "app_server.py"]
