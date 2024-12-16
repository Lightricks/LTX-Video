# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Install LTX-Video package and inference dependencies
RUN pip3 install . && \
    pip3 install accelerate matplotlib "imageio[ffmpeg]"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CKPT_DIR=/app/models
ENV OUTPUT_DIR=/app/outputs

# Create directories
RUN mkdir -p /app/models /app/outputs

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["python3", "api.py"]

# Add labels
LABEL maintainer="Lightricks" 
LABEL description="LTX-Video API service"
LABEL version="1.0"
