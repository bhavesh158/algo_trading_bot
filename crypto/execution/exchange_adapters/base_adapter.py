"""Abstract Exchange Adapter Interface.

All live exchange integrations must implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from crypto.core.models import Order


class BaseExchangeAdapter(ABC):
    """Interface for exchange API integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to exchange. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from exchange."""
        ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns the order with updated status."""
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
        """Get all open positions."""
        ...

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Get account balances. Returns {currency: amount}."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...
