# Slim, build-toolchain-free image. No TA-Lib, so no gcc/make/native builds.
FROM python:3.11-slim

LABEL maintainer="Andrew Schwartz <andrew@mk-dir.com>"

# uv: fast, reproducible dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies first so this layer caches across source changes.
COPY pyproject.toml ./
RUN uv sync --no-dev

# Application source.
COPY . .

# Default to paper live-trading; override at `docker run` to backtest/scan, e.g.
#   docker run --rm tradeflow uv run python main.py backtest
CMD ["uv", "run", "python", "main.py", "live"]
