"""Cancel all open orders on the configured Alpaca account.

Usage: ``python cancel_all_orders.py`` (or ``make cancel-orders``).
"""

import logging

from alpaca.trading.client import TradingClient

from src.brokers.alpaca.broker import AlpacaBroker
from src.settings import SettingsError, load_settings
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    try:
        settings = load_settings()
    except SettingsError as exc:
        raise SystemExit(str(exc))
    broker = AlpacaBroker(
        TradingClient(settings.alpaca_key, settings.alpaca_secret, paper=settings.paper_trade)
    )
    if broker.cancel_all_orders():
        logger.info("All open orders canceled.")
    else:
        logger.error("Failed to cancel orders.")


if __name__ == "__main__":
    main()
