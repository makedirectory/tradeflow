"""Deterministic order identity.

The guard against double-submitting used to be a question asked of the broker -
"are there open orders for this symbol?" - immediately before placing one. That is a
check-then-act race with a network round-trip in the middle, and it has no memory:
a process that restarts between submitting an order and seeing its fill asks the
question again, gets an answer that no longer reflects what it did, and submits the
same order a second time.

An identity fixes that without needing any memory at all. The same decision - this
strategy, this configuration, this symbol, this signal, on this bar - always hashes
to the same id, and a broker that has already accepted that id rejects the duplicate.
Idempotency becomes a property of the request rather than a property of how carefully
the caller looked first.

Stdlib only, and pure: trade-clock code, no I/O, no clock reads of its own.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Optional

#: Alpaca accepts client order ids up to 128 characters; 32 hex characters is far
#: inside that and still leaves collisions a non-issue at any plausible order rate.
_ID_LENGTH = 32

#: Config keys derived at construction rather than chosen by the caller. They are not
#: part of what makes a decision distinct, and including them would let an unrelated
#: internal change alter every order id.
_DERIVED_KEYS = frozenset({"required_lookback_periods"})


def strategy_fingerprint(strategy: Any) -> str:
    """A stable short identifier for a strategy *and the parameters it is running*.

    Two runs of the same class with different parameters are different strategies for
    this purpose: they can legitimately disagree about the same bar, and one must not
    be silently deduplicated against the other.
    """
    config = {
        key: value
        for key, value in sorted(getattr(strategy, "config", {}).items())
        if key not in _DERIVED_KEYS
    }
    # `default=str` so an unserializable value degrades to its repr rather than
    # raising inside the order path.
    payload = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{type(strategy).__name__}-{digest}"


def client_order_id(
    strategy: Any,
    symbol: str,
    signal: str,
    bar_timestamp: Optional[datetime] = None,
) -> str:
    """The id identifying one trading decision.

    ``bar_timestamp`` is what makes two otherwise identical decisions distinct: the
    same strategy re-entering the same symbol tomorrow is a genuinely new order, while
    the same bar replayed after a reconnect is not. When it is absent the id covers
    only the decision's content, which still collapses an immediate duplicate.
    """
    parts = [
        strategy_fingerprint(strategy),
        symbol,
        signal,
        bar_timestamp.isoformat() if bar_timestamp is not None else "",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:_ID_LENGTH]


def describe(strategy: Any, symbol: str, signal: str, bar_timestamp: Optional[datetime]) -> Dict[str, str]:
    """The inputs behind an id, for logging - so a duplicate can be explained."""
    return {
        "client_order_id": client_order_id(strategy, symbol, signal, bar_timestamp),
        "strategy": strategy_fingerprint(strategy),
        "symbol": symbol,
        "signal": signal,
        "bar": bar_timestamp.isoformat() if bar_timestamp is not None else "",
    }
