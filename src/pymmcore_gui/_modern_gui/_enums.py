from __future__ import annotations

from enum import Enum, auto


class DeviceStatus(Enum):
    """Connection status for the colored indicator dot."""

    CONNECTED = auto()
    DISCONNECTED = auto()
    BUSY = auto()
    ERROR = auto()
