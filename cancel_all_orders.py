"""Cancel all open orders on the configured Alpaca account.

Usage: ``python cancel_all_orders.py`` (or ``make cancel-orders``).
"""

import logging

from alpaca.trading.client import TradingClient

import config
from src.brokers.alpaca.broker import AlpacaBroker
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    broker = AlpacaBroker(
        TradingClient(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY, paper=config.PAPER_TRADE)
    )
    if broker.cancel_all_orders():
        logger.info("All open orders cancelled.")
    else:
        logger.error("Failed to cancel orders.")


if __name__ == "__main__":
    main()
