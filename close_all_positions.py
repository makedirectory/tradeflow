"""Liquidate all open positions on the configured Alpaca account.

Usage:
    python close_all_positions.py                  # also cancels open orders
    python close_all_positions.py --keep-orders    # leave open orders in place
"""

import argparse
import logging

from src.brokers.alpaca.factory import build_broker
from src.settings import SettingsError, load_settings
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Close all open positions")
    parser.add_argument("--keep-orders", action="store_true", help="Do not cancel open orders first")
    args = parser.parse_args()

    try:
        settings = load_settings()
    except SettingsError as exc:
        raise SystemExit(str(exc))
    broker = build_broker(settings.alpaca_key, settings.alpaca_secret, settings.paper_trade)
    if broker.close_all_positions(cancel_orders=not args.keep_orders):
        logger.info("All positions closed.")
    else:
        logger.error("Failed to close positions.")


if __name__ == "__main__":
    main()
