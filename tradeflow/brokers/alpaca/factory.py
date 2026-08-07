"""Factories that build Alpaca-backed brokers and market-data providers.

Callers above the broker layer (the CLI, services, account scripts) construct
Alpaca objects exclusively through these functions, so the ``alpaca`` SDK is
never imported outside :mod:`tradeflow.brokers.alpaca`.
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from tradeflow.brokers.alpaca.broker import AlpacaBroker
from tradeflow.brokers.alpaca.market_data import AlpacaMarketData


def build_market_data(api_key: str, api_secret: str) -> AlpacaMarketData:
    """Build the historical/live market-data provider.

    Constructs no trading client: a process holding only this object is
    structurally incapable of placing orders.
    """
    historical = StockHistoricalDataClient(api_key, api_secret)
    return AlpacaMarketData(historical, api_key, api_secret)


def build_broker(api_key: str, api_secret: str, paper: bool = True) -> AlpacaBroker:
    """Build the trading broker (account, orders, positions, trade updates)."""
    trading_client = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)
    return AlpacaBroker(trading_client, api_key, api_secret, paper)
