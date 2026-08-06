# Slim, build-toolchain-free image. No TA-Lib, so no gcc/make/native builds.
FROM python:3.11-slim

LABEL maintainer="Andrew Schwartz <andrew@mk-dir.com>"

# uv: fast, reproducible dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Which optional capabilities this image carries. `uv sync --no-dev` would exclude
# them all, leaving the MCP server and the bar cache unable to import — so the
# services that need them are carried by default. Paid credentials are not: the
# research agent's `ai` extra is opt-in via `--build-arg EXTRAS="mcp,store,ai"`,
# because the base image should need zero paid API keys to be useful.
ARG EXTRAS="mcp,store,viz"

# Install dependencies first so this layer caches across source changes.
COPY pyproject.toml README.md ./
COPY tradeflow ./tradeflow
RUN uv sync --no-dev $(echo "$EXTRAS" | tr ',' '\n' | sed 's/^/--extra /' | tr '\n' ' ')

# Application source.
COPY . .

# State lives on a mounted volume, never in the container layer. Without this a
# containerized research session journals its trials into a filesystem that
# evaporates on `docker compose down` — silently resetting the campaign count that
# every multiple-testing correction depends on.
ENV TRADEFLOW_HOME=/state

# A non-root user, with the state directories owned by it. The uid is pinned so a
# host user can read the same volume without sudo: mixed native/container workflows
# against one journal are certain, and a root-owned journal breaks them.
#
# Each mount point must exist *in the image* and be owned correctly, because Docker
# seeds a named volume's ownership from the image directory at that path — and
# creates it root-owned when there is nothing there to copy. Creating only /state
# leaves every subdirectory below it unwritable by this user, which surfaces as
# "unable to open database file" the first time anything journals.
RUN useradd --uid 1000 --create-home tradeflow \
    && mkdir -p /state/logs /state/cache /state/configs \
    && chown -R tradeflow:tradeflow /state /app
USER tradeflow

# A safe default. This image previously defaulted to `live`, so `docker run` with no
# arguments started a paper-trading loop — turning the machine on must not turn
# trading on. Override with any verb:
#   docker run --rm tradeflow uv run python main.py demo
CMD ["uv", "run", "python", "main.py", "--help"]
