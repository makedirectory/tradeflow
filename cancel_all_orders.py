"""Cancel all open orders on the configured Alpaca account.

Usage: ``python cancel_all_orders.py`` (or ``make cancel-orders``).
"""

import logging

from tradeflow.brokers.alpaca.factory import build_broker
from tradeflow.settings import SettingsError, load_settings
from tradeflow.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    try:
        settings = load_settings()
    except SettingsError as exc:
        raise SystemExit(str(exc))
    broker = build_broker(settings.alpaca_key, settings.alpaca_secret, settings.paper_trade)
    if broker.cancel_all_orders():
        logger.info("All open orders canceled.")
    else:
        logger.error("Failed to cancel orders.")


if __name__ == "__main__":
    main()
