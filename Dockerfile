FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
# `sentence-transformers` depends on `torch`; PyPI's default `torch` wheel
# bundles the full NVIDIA CUDA toolkit (multiple GB) even though this
# service only ever runs on CPU (prompt #72 -- "CPU execution is
# acceptable... no GPU requirement"). Installing the CPU-only build from
# PyTorch's own index first means the normal install below finds it
# already satisfied and never pulls the GPU variant -- image size and
# build time both drop by several GB with identical runtime behavior.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

# Copy application source.
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
