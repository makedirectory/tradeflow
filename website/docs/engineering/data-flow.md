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
Strategy.generate_signals(processed)      # {timestamp: BUY/SELL/HOLD/...}
        │
        ▼
BacktestEngine._simulate_symbol(...)      # bar-by-bar fills, stop/take/exit
        │  trades
        ▼
analytics.performance.compute_backtest_metrics(...)   # -> metrics dict
analytics.reporting.log_backtest_report(...)          # -> text
```

The engine carries realised cash across symbols so a run can't spend the same
dollar twice, and force-closes any open position at the final bar.

## Live

```
LiveEngine.start(universe)
   ├── _warm_up: get_bars(lookback) -> process_data -> strategy.warm_up(buffer)
   └── MarketDataClient.stream(universe, on_bar)
            │ BarEvent per bar
            ▼
        Strategy.process_real_time_data(symbol, price, volume, ts)  # -> signal
            │ actionable signal
            ▼
        LiveTrader.handle_signal(symbol, signal, price)
            ├── entry: size via Strategy.calculate_position_size -> submit_bracket_order
            └── exit:  Broker.close_position
            │
            ▼
        AlpacaBroker -> Alpaca SDK
```

## Why the pipeline is shared

`process_data` and `generate_signals` are identical in both modes — the strategy
doesn't know whether it's being backtested or traded live. The difference is only
*what consumes the signal*: a simulator (engine) or a broker (execution). This is
the payoff of [separating concerns](separation-of-concerns).
