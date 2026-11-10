---
sidebar_position: 14
title: Running in Docker
---

# Running in Docker

```bash
make up          # docker compose up -d — MCP + persistent state
make down        # stop; your trial history survives
```

**This is a local development stack, not an easier way to try QuantRed.** Docker is
itself a prerequisite, so if you just want to run this, [installing the
command](installation#as-a-command-no-clone) is strictly simpler:

```bash
uv tool install quantred && quantred demo
```

What compose adds is the volume and environment wiring people get wrong by hand, a
long-running MCP server and docs site, and the seam a database service would attach
to later without changing how anything boots.

Honest framing: QuantRed is a single-process application with file-backed state, so
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
docker compose run --rm verdict --scanner none --symbols AAPL,MSFT,NVDA,AMZN,GOOGL,META,AVGO,JPM,XOM,UNH,COST,WMT --start 2024-01-01 --end 2024-12-31
docker compose run --rm backtest --symbols NVDA --start 2024-01-01 --end 2024-06-01
docker compose run --rm trials best
```

Or via the Makefile: `make compose-run CMD="verdict --scanner none"`.

The documentation site runs under its own profile:

```bash
docker compose --profile docs up docs      # http://localhost:3000
```

## State, and why the volumes matter

Three named volumes hold everything the application writes — the research journal
and trial store (`/state/logs`), the bar cache (`/state/cache`), and promoted
configs (`/state/configs`). The host's state-root `.env`
(`~/.quantred/.env`, or `$QUANTRED_HOME/.env` when set on the host) is bind-mounted
**read-only** at `/state/.env`, which is where the image reads credentials
(`QUANTRED_HOME=/state`). A container should not be able to rewrite the keys it was
given, so keys go in that host file — `quantred init` on the host writes it — not through
`init` inside the container (the container's own messages say so). Create it before
the first `docker compose run`: Docker turns a missing bind-mount source into an empty
directory.

Nothing else from the checkout reaches the image: `.dockerignore` is an allowlist of
the package, the example pack and the project metadata, so `.env`, `specs/`, `configs/`,
`logs/`, `cache/` and `.git` stay out of the build context. The image's environment is
built once with the extras it carries and is not re-synced at start, so a container
runs without network access.

This is the part worth getting right. A compose setup that looks fine while
journaling trials into the container layer would discard them on `down`, silently
resetting the campaign count that every multiple-testing correction depends on — and
nothing would error. `scripts/compose_smoke.sh` tests exactly that: journal a trial (no keys needed),
`down`, and check the trial is still there.

**Names.** The compose project is pinned as `quantred` and the volumes are
`quantred-logs`, `quantred-cache` and `quantred-configs`, so Docker sees them as
`quantred_quantred-logs` and so on. The project name is pinned because an unpinned one
follows the checkout directory, and a second clone under another name would attach to
new, empty volumes instead of the journal the first one wrote.

**Coming from the old names.** Before the rename these were `tradeflow_tradeflow-logs`,
`tradeflow_tradeflow-cache` and `tradeflow_tradeflow-configs`. A renamed volume is a new
empty one and nothing copies the old ones for you, so on a machine that ran the stack
under the old names, stop it and copy each volume across once:

```bash
docker compose down
for v in logs cache configs; do
  docker run --rm -v tradeflow_tradeflow-$v:/from -v quantred_quantred-$v:/to alpine cp -a /from/. /to/
done
```

`docker volume ls` shows whether the old ones exist. Until they are copied, the stack
starts with an empty journal: every earlier trial is missing from the campaign count,
and the old volumes are untouched, so nothing is lost by copying late.

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
{"mcpServers": {"quantred": {"command": "docker",
  "args": ["compose", "-f", "/path/to/quantred/docker-compose.yml", "run", "--rm", "mcp"]}}}
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
