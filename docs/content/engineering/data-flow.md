---
sidebar_position: 4
title: Data flow
---

# Data flow

Both modes share the same conceptual pipeline; only the last stage differs
(simulate vs. execute).

## Backtest

```
SymbolScanner.scan(candidates)            # marketdata -> scan signals
        │  universe
        ▼
MarketDataClient.get_bars(universe, tf, start, end)   # {symbol: OHLCV}
        │
        ▼  per symbol
Strategy.process_data(bars)               # + indicator columns
Strategy.calculate_scores(processed)      # {timestamp: signed score}  ← the one decision
Strategy.generate_signals(processed)      # base class derives {timestamp: BUY/SELL/HOLD/...}
        │
        ▼
BacktestEngine._replay(...)               # one merged timeline, one capital pool
                                          # mark -> exit -> rank -> admit -> record
        │  trades
        ▼
analytics.performance.compute_backtest_metrics(...)   # -> metrics dict
analytics.reporting.log_backtest_report(...)          # -> text
```

Every symbol is simulated on a single merged timeline against one shared capital
pool, so positions compete for the same dollars as they would live: exits settle
before entries within a bar, and simultaneous candidates are admitted in
conviction order while position limits and free cash allow. Any position still
open at the final bar is force-closed. See
[portfolio accounting](engine.md#portfolio-accounting).

## Live

```
LiveEngine.start(universe)
   ├── _warm_up:   get_bars(sessions) -> process_data -> strategy.warm_up(buffer)
   ├── _cold_start: list_positions -> hydrate the strategy's position book
   └── MarketDataClient.stream(universe, on_bar)   # auto-reconnecting bar stream
            │ BarEvent (full OHLCV) per bar
            ▼
        BarQualityFilter.check(...)          # reject, never repair
            │ accepted bars only
            ▼
        Strategy.process_bar(symbol, {o,h,l,c,v}, ts)   # -> signal
            │ actionable signal
            ▼
        LiveTrader.handle_signal(symbol, signal, price, bar_timestamp)
            ├── halted? -> Decision(allowed=False)   # entries only; exits pass
            ├── entry: PositionSizer.size(...) -> submit_bracket_order(client_order_id)
            └── exit:  Broker.close_position
            │
            ▼ Decision(allowed, reason, guards_consulted, order)
        PositionLedger.record_decision(...)   # including the declines
            │
            ▼
        AlpacaBroker -> Alpaca SDK
```

The stream feeds the strategy the **full streamed OHLCV bar** (not just the close),
and the `PositionSizer` is pluggable — `RiskBasedSizer` by default, or
`PortfolioWeightSizer` when running `live --portfolio`. See
[Portfolio manager](portfolio).

## Why the pipeline is shared

`process_data` and `generate_signals` are identical in both modes — the strategy
doesn't know whether it's being backtested or traded live. The difference is only
*what consumes the signal*: a simulator (engine) or a broker (execution). This is
the payoff of [separating concerns](separation-of-concerns).
