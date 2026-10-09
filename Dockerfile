FROM nvidia/cuda:13.1.1-devel-ubuntu22.04

# Avoid interaction during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 (matches the frozen requirements) and build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Symlink python to point to 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

WORKDIR /app

# Copy exact frozen requirements (see header in requirements.txt)
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip
# torch first, CUDA build from the PyTorch index; the requirements pin
# (torch==2.14.1) is then already satisfied and kept as-is.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1
RUN python -m pip install --no-cache-dir -r requirements.txt

# Register CUDA libraries for bitsandbytes
RUN echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig
RUN rm -rf /usr/local/cuda/compat

# Project setup
COPY . .
ENV PYTHONPATH=/app

# Execute as a module
CMD ["python", "-m", "src.main"]
