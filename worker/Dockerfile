FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (cached unless pyproject.toml changes)
COPY pyproject.toml README.md ./
RUN mkdir -p src/harkd && echo '__version__ = "0.0.0"' > src/harkd/__init__.py && \
    pip install --no-cache-dir --break-system-packages . && \
    rm -rf src

# Copy real source and install package only (deps already cached)
COPY src/ src/
RUN pip install --no-cache-dir --no-deps --break-system-packages .

ENV HARKD_WORKER_MODEL=large-v3
ENV HARKD_WORKER_DEVICE=auto
ENV HARKD_WORKER_API_KEY=""
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "harkd.worker.app:app", "--host", "0.0.0.0", "--port", "8000"]
