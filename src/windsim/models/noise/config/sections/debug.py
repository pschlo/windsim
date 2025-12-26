from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._abstract import Section


@dataclass
class BarrierFilter:
    postcode: str | None = None

@dataclass
class JitterBuildingHeights:
    average_offset: float
    max_offset_deviation: float

@dataclass
class JitterTurbinePositions:
    max_offset_distance: float


class DebugSection(Section):
    random_seed: int | None
    jitter_turbine_positions: JitterTurbinePositions | None
    jitter_building_heights: JitterBuildingHeights | None
    discard_invalid_turbines: bool
    filter_barriers: BarrierFilter | None

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        if 'random_seed' not in raw['enabled']:
            self.random_seed = None
        else:
            self.random_seed = raw['random_seed']

        if 'jitter_turbine_positions' not in raw['enabled']:
            self.jitter_turbine_positions = None
        else:
            self.jitter_turbine_positions = JitterTurbinePositions(**raw['jitter_turbine_positions'])

        if 'jitter_building_heights' not in raw['enabled']:
            self.jitter_building_heights = None
        else:
            self.jitter_building_heights = JitterBuildingHeights(**raw['jitter_building_heights'])

        self.discard_invalid_turbines = 'discard_invalid_turbines' in raw['enabled']

        if 'filter_barriers' not in raw['enabled']:
            self.filter_barriers = None
        else:
            self.filter_barriers = BarrierFilter(**raw['filter_barriers'])
