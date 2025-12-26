from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ConfigError
from ._abstract import Section


@dataclass
class MapTilesConfig:
    api_key: str
    zoom: int
    style: str = 'alidade_bright'
    double_resolution: bool = False


class OutputMapSection(Section):
    buffer: float
    indicate_input_area: bool
    contour_line_visibility: float
    contour_fill_visibility: float
    tiles: MapTilesConfig
    contour_levels: float | list[float]
    add_contour_labels: bool
    add_receiver_labels: bool
    individual_colors: bool
    use_computation_crs: bool

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        self.buffer = raw.get('buffer', 0)
        self.indicate_input_area = raw['indicate_input_area']
        self.contour_line_visibility = raw.get('contour_line_visibility', 0)
        self.contour_fill_visibility = raw.get('contour_fill_visibility', 0)
        self.individual_colors = raw['individual_colors']
        self.use_computation_crs = raw['use_computation_crs']
        self.tiles = MapTilesConfig(**raw['tiles'])

        r = raw['contour_levels']
        if isinstance(r, dict):
            self.contour_levels = list(np.arange(r['start'], r['stop'] + r['step']/2, r['step']))
        else:
            self.contour_levels = r

        self.add_contour_labels = raw['add_contour_labels']
        self.add_receiver_labels = raw['add_receiver_labels']


class OutputFileSection(Section):
    folder: Path

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        self.folder = Path(raw['folder']).resolve()
        self.folder.mkdir(exist_ok=True)
        if not self.folder.exists():
                raise ConfigError(f"Output folder '{self.folder}' does not exist")


class OutputSection(Section):
    map: OutputMapSection | None
    file: OutputFileSection | None

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        if 'map' in raw['assets']:
            self.map = OutputMapSection(raw['map'])
        else:
            self.map = None

        if 'file' in raw['assets']:
            self.file = OutputFileSection(raw['file'])
        else:
            self.file = None

