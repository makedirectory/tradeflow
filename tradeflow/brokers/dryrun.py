"""A broker that can answer questions and cannot trade.

Observing a strategy's decision path used to mean running a paper session, and a paper
session needed fills to observe — so the book's caps got distorted downward to force
them. A position ceiling small enough to guarantee fills biases the book toward
low-priced names and turns every high-price signal into an invisible non-trade, which
means the sample you collect is not the strategy you validated. Measuring execution
required trading a book nobody wanted to trade.

This separates the two questions. **What would this contract try to do?** is answered
here, against a stated account, with the caps exactly as configured. **What does
execution actually cost?** is a different run at reduced capital, where the real
contract's proportions are preserved and fills are the point.

**Trading is a capability this broker does not have.** Every order method refuses, and
that is the whole design rather than a flag consulted on the order path. A flag can be
forgotten on one branch, read from a stale config, or inverted in a refactor; an absent
capability cannot be any of those. It is the same guarantee the MCP server gets from
building only a data client — the wall is structural, so it cannot be reasoned around.

Reads are served from a synthetic account whose capital the caller must state. There is
deliberately no fallback to a broker's real equity and no default account size: the
question is what a *specified* contract would do at a *specified* capital, and inventing
either turns the answer into a guess about a book nobody chose.
"""

from typing import Any, Dict, List, Optional

from tradeflow.brokers.base import AccountSnapshot, Broker, MarketStatus, OrderResult, Position
from tradeflow.brokers.errors import BrokerError

#: Printed wherever a dry run reports, and deliberately unmissable. A reader who mistakes
#: this for a paper session mistakes an intention for an execution.
BANNER = "DRY RUN — broker has no trading capability; no orders can be submitted"

#: Raised by every order method. The message names the mode rather than the method,
#: because a caller reaching one of these has misunderstood the run, not the API.
_REFUSAL = (
    "this is a dry run: the broker has no trading capability, so nothing can be "
    "submitted, cancelled or closed. Use a small-real session to observe execution"
)


class DryRunTradingAttempted(BrokerError):
    """Raised wherever a dry run reaches an order path.

    A :class:`~tradeflow.brokers.errors.BrokerError` on purpose, and the choice is the
    whole design. ``LiveTrader`` builds its :class:`OrderPlan` *before* submitting
    precisely so that a broker refusal still records what would have been sent, and
    returns a decline carrying that plan. So refusing here does not lose the intent —
    it routes it through machinery the trade clock already has, and the report is
    assembled from real declines rather than from a parallel path that would have to be
    kept in step with the real one.

    The alternative was a broker that accepts submissions and records them. That would
    make "cannot trade" a promise about what this class does with an order rather than
    about whether it can place one, and the guarantee this mode exists for is the
    second.
    """


class DryRunBroker(Broker):
    """Answers account and market questions; refuses every order.

    ``positions`` start empty unless the caller supplies them. A dry run over an
    existing book and a dry run from flat answer different questions - "what would this
    do next, given what I hold" against "what would this do from a clean book" - so the
    source is recorded rather than assumed, and nothing is adopted implicitly.
    """

    def __init__(
        self,
        *,
        capital: float,
        positions: Optional[List[Position]] = None,
        positions_source: str = "flat",
        tradable: Optional[Dict[str, bool]] = None,
        market_open: bool = True,
    ):
        if capital is None:
            raise ValueError(
                "a dry run needs a stated capital: take it from the config being run or "
                "pass one explicitly. There is no default, because the caps a run reports "
                "are only meaningful against the capital they bound"
            )
        self._capital = float(capital)
        self._positions = list(positions or [])
        #: Where the starting book came from, carried into the report so a reader is
        #: never left guessing whether an empty book was chosen or merely assumed.
        self.positions_source = positions_source
        self._tradable = dict(tradable or {})
        self._market_open = bool(market_open)
        #: Everything the decision path *would* have sent, in order. Named for what it
        #: is: these are intentions, and no word here may suggest an execution.
        self.would_submit: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Reads — answerable without a venue
    # ------------------------------------------------------------------ #
    def get_account(self) -> Optional[AccountSnapshot]:
        """The stated account, marked at the positions it was given.

        ``buying_power`` is the cash the sizer may deploy, so it is capital less what
        the supplied positions already reserve - not the capital again, which would let
        a dry run over an existing book size as though it were flat.
        """
        reserved = sum(p.market_value for p in self._positions)
        return AccountSnapshot(
            cash=self._capital - reserved,
            equity=self._capital,
            buying_power=max(self._capital - reserved, 0.0),
            portfolio_value=self._capital,
            trading_blocked=False,
        )

    def list_positions(self) -> List[Position]:
        return list(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        return next((p for p in self._positions if p.symbol == symbol), None)

    def is_tradable(self, symbol: str) -> bool:
        """Assumed tradable unless the caller said otherwise.

        A dry run cannot ask a venue, and refusing every symbol would report a book that
        skipped everything for a reason that is an artefact of the mode.
        """
        return self._tradable.get(symbol, True)

    def list_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Always empty. Nothing was submitted, so nothing can be resting."""
        return []

    def get_market_status(self) -> Optional[MarketStatus]:
        return MarketStatus(is_open=self._market_open)

    def supports_trade_updates(self) -> bool:
        """No. There are no fills to stream, and saying otherwise would have a caller
        wait on updates that can never arrive."""
        return False

    # ------------------------------------------------------------------ #
    # Orders — the capability this broker does not have
    # ------------------------------------------------------------------ #
    def submit_market_order(self, *args: Any, **kwargs: Any) -> OrderResult:
        raise DryRunTradingAttempted(_REFUSAL)

    def submit_bracket_order(self, *args: Any, **kwargs: Any) -> OrderResult:
        raise DryRunTradingAttempted(_REFUSAL)

    def cancel_order(self, order_id: str) -> bool:
        raise DryRunTradingAttempted(_REFUSAL)

    def cancel_all_orders(self) -> bool:
        raise DryRunTradingAttempted(_REFUSAL)

    def close_position(self, symbol: str) -> bool:
        raise DryRunTradingAttempted(_REFUSAL)

    def close_all_positions(self, cancel_orders: bool = True) -> bool:
        raise DryRunTradingAttempted(_REFUSAL)

    async def stream_trade_updates(self, handler: Any) -> None:
        raise DryRunTradingAttempted(_REFUSAL)
