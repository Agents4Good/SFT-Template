FROM nvidia/cuda:13.1.1-devel-ubuntu22.04

# Avoid interaction during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.13 and essential build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Symlink python to point to 3.13
RUN ln -sf /usr/bin/python3.13 /usr/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3

# Install pip for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /app

# Copy exact local requirements
COPY requirements.txt .

# --- LOCAL REPLICATION FIX ---
# Since your local environment works with conflicting versions (torch 2.11 vs unsloth < 2.11),
# we install dependencies first, then unsloth WITHOUT dependency checking.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN grep -v "unsloth" requirements.txt > req_no_unsloth.txt
RUN python -m pip install --no-cache-dir -r req_no_unsloth.txt
RUN python -m pip install --no-cache-dir --no-deps unsloth==2026.4.4 unsloth_zoo==2026.4.6

# --- THE CLEAN FIX ---
# Register CUDA libraries for bitsandbytes
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig

# Project setup
COPY . .
ENV PYTHONPATH=/app

# Execute as a module
CMD ["python", "-m", "src.main"]
