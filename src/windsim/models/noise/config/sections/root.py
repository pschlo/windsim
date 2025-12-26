from __future__ import annotations

from typing import Any

import numpy as np

from ._abstract import Section
from .computation import ComputationSection
from .debug import DebugSection
from .input import InputSection
from .output import OutputSection


class ConfigData(Section):
    """The root section."""
    time_decimal_places: int
    rng: np.random.Generator

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        self.time_decimal_places = 2
        
        self.computation = ComputationSection(raw['computation'])
        self.input = InputSection(raw['input'])
        self.output = OutputSection(raw['output'])
        self.debug = DebugSection(raw['debug'])

        self.rng = np.random.default_rng(self.debug.random_seed)
