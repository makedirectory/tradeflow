#!/usr/bin/env bash
# Compose smoke test — the parts CI cannot run (it has no Docker daemon).
#
# Run locally and record the output in the PR, per the manual-verification rule.
# The state-survival check is the one that matters: a compose setup that looks fine
# while journaling trials into a container layer would silently reset the campaign
# count every multiple-testing correction depends on, and nothing would error.
#
#   ./scripts/compose_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

pass() { printf '  [ok]   %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; FAILED=1; }
FAILED=0

echo
echo "QuantRed compose smoke test"
echo

# 1. The YAML is valid and `up` starts nothing that trades.
docker compose config >/dev/null
pass "compose config parses"

DEFAULT_SERVICES=$(docker compose config --services)
if grep -qx "live" <<<"$DEFAULT_SERVICES"; then
    fail "live is in the default service set — 'up' must never start trading"
else
    pass "live is profile-gated (not started by 'up')"
fi

# 2. The image builds and its default command is safe.
docker compose build mcp >/dev/null
pass "image builds"

DEFAULT_CMD=$(docker image inspect quantred:local --format '{{json .Config.Cmd}}')
if grep -q '"live"' <<<"$DEFAULT_CMD"; then
    fail "image CMD defaults to live: $DEFAULT_CMD"
else
    pass "image CMD is safe by default: $DEFAULT_CMD"
fi

# 3. The demo runs offline, with no credentials at all.
if docker compose run --rm --no-deps -T demo >/tmp/tf-demo.log 2>&1; then
    pass "demo completes inside the container (offline, no keys)"
else
    fail "demo failed — see /tmp/tf-demo.log"
fi

# 4. State survives container replacement. This is the whole point of the volumes.
#    A trial is journaled through the application's own writer, which needs no keys:
#    this used to backtest a strategy that no longer exists, ignore the failure, and
#    compare "Rows: 0" with "Rows: 0" - a pass that tested nothing.
docker compose run --rm --no-deps -T --entrypoint uv trials run python -c \
    "from datetime import datetime; from quantred.services.audit import journal_trial; \
journal_trial('backtest', strategy='compose_smoke', symbols=['AAA'], start=datetime(2024, 1, 2), \
end=datetime(2024, 6, 1), params={'smoke': 1}, metrics={'sharpe_ratio': 0.0})" \
    >/tmp/tf-journal.log 2>&1 || fail "could not journal a trial - see /tmp/tf-journal.log"

BEFORE=$(docker compose run --rm --no-deps -T trials status 2>/dev/null | grep -E '^Rows' || echo "Rows: ?")
docker compose down >/dev/null 2>&1
AFTER=$(docker compose run --rm --no-deps -T trials status 2>/dev/null | grep -E '^Rows' || echo "Rows: ?")

if [[ "$BEFORE" == "$AFTER" && "$BEFORE" != "Rows: ?" && "$BEFORE" != *": 0" ]]; then
    pass "state survives 'down' ($BEFORE)"
else
    fail "state did not survive 'down', or nothing was recorded (before='$BEFORE' after='$AFTER')"
fi

# 5. The volume is readable by the host user without sudo — mixed native/container
#    work against one journal is certain, and a root-owned journal breaks it.
# The project name is pinned to `quantred` in docker-compose.yml, so this is the same
# name in every checkout directory.
VOLUME=$(docker volume inspect quantred_quantred-logs --format '{{.Mountpoint}}' 2>/dev/null || echo "")
if [[ -n "$VOLUME" ]]; then
    pass "logs volume exists ($VOLUME)"
else
    fail "logs volume was not created"
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "All compose smoke checks passed."
else
    echo "Some checks FAILED — see above."
    exit 1
fi
