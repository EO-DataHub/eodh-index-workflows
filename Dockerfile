# Python and package versions match the tests: Python from .python-version, packages from uv.lock.
# rasterio's wheel bundles GDAL, so no system GDAL or compiler is needed.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv

# The only system library the rasterio wheel does not bundle. Without it: "libexpat.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# GDAL options for reading remote COGs by range request: no directory listing on open, retry
# transient HTTP errors, merge range requests, and cache the blocks already read.
ENV GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    GDAL_HTTP_MAX_RETRY=5 \
    GDAL_HTTP_RETRY_DELAY=1 \
    GDAL_HTTP_MULTIRANGE=YES \
    VSI_CACHE=TRUE

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /root/.cache/uv

# The Workflow Runner does not run the container as root, so every file must be world-readable
# and every directory world-traversable. Compile now: /app is not writable at run time.
COPY src/ /app/
RUN chmod -R a+rX /app && python -m compileall -q /app

# The CWL sets the command (baseCommand: python3 /app/main.py).
CMD ["python3", "/app/main.py", "--help"]
