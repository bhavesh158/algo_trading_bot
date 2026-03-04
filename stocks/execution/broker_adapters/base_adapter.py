"""Abstract Broker Adapter Interface.

All live broker integrations must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from stocks.core.models import Order


class BaseBrokerAdapter(ABC):
    """Interface for broker API integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker API. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker API."""
        ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker. Returns the order with updated status."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True on success."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order | None:
        """Get current status of an order."""
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Get all open positions from the broker."""
        ...

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get current account balance."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if broker connection is active."""
        ...
