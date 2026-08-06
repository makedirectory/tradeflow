---
sidebar_position: 14
title: Running in Docker
---

# Running in Docker

```bash
make up          # docker compose up -d — MCP + persistent state
make down        # stop; your trial history survives
```

**This is a local development stack, not an easier way to try TradeFlow.** Docker is
itself a prerequisite, so if you just want to run this, [installing the
command](installation#as-a-command-no-clone) is strictly simpler:

```bash
uv tool install git+https://github.com/makedirectory/tradeflow && tradeflow demo
```

What compose adds is the volume and environment wiring people get wrong by hand, a
long-running MCP server and docs site, and the seam a database service would attach
to later without changing how anything boots.

Honest framing: TradeFlow is a single-process application with file-backed state, so
most of the compose file is volumes and named entrypoints rather than service
orchestration.

## `up` never starts trading

The default `up` boots research-clock surfaces only. `live` sits behind its own
profile and is not in the default service set — turning the machine on must not turn
trading on, not even paper trading:

```bash
docker compose run --rm live      # the only way to start it
```

The image's default command is `--help`, not a trading loop. `PAPER_TRADE` is still
asserted inside the container, so the profile is the outer of two locks rather than
a replacement for one.

## One-shot commands

Same image, same state, different verb — arguments pass straight through:

```bash
docker compose run --rm demo
docker compose run --rm verdict --symbols NVDA,AAPL,META --start 2024-01-01 --end 2024-12-31
docker compose run --rm backtest --symbols NVDA --start 2024-01-01 --end 2024-06-01
docker compose run --rm trials best
```

Or via the Makefile: `make compose-run CMD="verdict --symbols NVDA"`.

The documentation site runs under its own profile:

```bash
docker compose --profile docs up docs      # http://localhost:3000
```

## State, and why the volumes matter

Three named volumes hold everything the application writes — the research journal
and trial store (`/state/logs`), the bar cache (`/state/cache`), and promoted
configs (`/state/configs`). `.env` is bind-mounted **read-only** from the host: a
container should not be able to rewrite the keys it was given.

This is the part worth getting right. A compose setup that looks fine while
journaling trials into the container layer would discard them on `down`, silently
resetting the campaign count that every multiple-testing correction depends on — and
nothing would error. `scripts/compose_smoke.sh` tests exactly that: run a backtest,
`down`, and check the trial is still there.

**Ownership.** The image runs as a pinned non-root uid (1000) and creates each mount
point *in the image* so the named volumes inherit that ownership. Docker seeds a
volume's ownership from the image directory at that path and makes it root-owned
when nothing is there — which surfaces as "unable to open database file" the first
time anything journals. Because the uid is pinned, a host user can read the same
volume without `sudo`, which mixed native/container work against one journal
requires.

## Connecting an MCP client

The server speaks stdio, so the `mcp` service is attach-oriented rather than a
listening port:

```json
{"mcpServers": {"tradeflow": {"command": "docker",
  "args": ["compose", "-f", "/path/to/tradeflow/docker-compose.yml", "run", "--rm", "mcp"]}}}
```

A TCP/SSE transport would add a port mapping and nothing else; it is not built.

## Optional extras in the image

The image carries `mcp`, `store`, and `viz` by default. The research agent's `ai`
extra is opt-in, because the base image should need zero paid credentials:

```bash
docker compose build --build-arg EXTRAS="mcp,store,viz,ai"
```

## Verifying the wiring

CI validates that the compose file parses (cheap, needs no daemon). The behavioral
checks need Docker and are a script you run locally:

```bash
make compose-smoke
```

It checks that the config parses, that `live` is not in the default service set,
that the image's default command is safe, that the demo runs offline with no keys,
that state survives `down`, and that the volumes exist where the host can reach them.
