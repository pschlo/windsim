from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict, cast


class Section(ABC):
    @abstractmethod
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
