FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp using CMake
WORKDIR /app/llama.cpp
RUN mkdir build && cd build && cmake .. -DLLAMA_CURL=OFF -DLLAMA_BLAS=OFF -DLLAMA_NATIVE=OFF && make -j$(nproc)

# Install Python dependencies
RUN pip3 install fastapi uvicorn requests

# Copy server script
COPY server.py /app/llama.cpp/

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8080

# Start the server (model will be downloaded at runtime if not present)
CMD ["python3", "server.py", "--model", "models/ggml-model-q4_0.gguf", "--host", "0.0.0.0", "--port", "8080"] 