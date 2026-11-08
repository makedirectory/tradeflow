"""Broker failure, typed.

Every broker call used to fail the same way: ``None``, or ``False``. A rate limit, an
expired token, insufficient buying power, a market that is closed, and an order the
venue deliberately refused all arrived as the same absence of information, so the only
possible response was the same one - log it and move on. That is the wrong response to
most of them, and for a duplicate order it is actively misleading: the order *was*
placed.

The distinctions that matter are the ones a caller would act on differently:

* :class:`RateLimitedError` - back off; the request was fine.
* :class:`AuthenticationError` - nothing will work until a human fixes credentials.
  Never transient, so never worth retrying or trading through.
* :class:`InsufficientFundsError` - the account cannot support this size.
* :class:`MarketClosedError` - correct request, wrong time.
* :class:`DuplicateOrderError` - the venue already has this order. Not a failure.
* :class:`OrderRejectedError` - the venue refused it on its own terms.
* :class:`BrokerUnavailableError` - the venue could not be reached at all. Transient
  by nature, and the only one where carrying on optimistically is defensible.

Vendor-neutral by design: these live beside the :class:`~tradeflow.brokers.base.Broker`
interface, so a caller that handles them handles every venue.
"""


class BrokerError(RuntimeError):
    """Base class for every broker failure."""


class BrokerUnavailableError(BrokerError):
    """The venue could not be reached (network failure, timeout, 5xx)."""


class RateLimitedError(BrokerError):
    """The venue is throttling us. The request itself was acceptable."""


class AuthenticationError(BrokerError):
    """Credentials are missing, expired, or insufficient for this operation.

    Distinguished from :class:`BrokerUnavailableError` because it is never transient:
    retrying cannot fix it, and continuing to trade on the assumption that things are
    fine is worse than stopping.
    """


class InsufficientFundsError(BrokerError):
    """Buying power or position size cannot support the request."""


class MarketClosedError(BrokerError):
    """The venue rejected the order because the market is not open."""


class OrderRejectedError(BrokerError):
    """The venue refused the order on validation or compliance grounds."""


class DuplicateOrderError(BrokerError):
    """This client order id has already been accepted.

    Raised rather than returned because it interrupts the caller's flow - but it
    reports a *success*: the order exists at the venue. A caller must not treat it as
    a failed submission and must not resubmit.
    """


class NotTradableError(BrokerError):
    """The symbol cannot be traded on this venue right now."""
