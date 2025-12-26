from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from noise_simulation.coordinate_reference_systems import CRS, parse_name as parse_crs_name
from .. import area_specs
from ..exceptions import ConfigError
from ._abstract import Section


@dataclass
class GridReceivers:
    spacing: float
    height: float
    constant_elevation: float | None = None

@dataclass
class NormalReceivers:
    default_height: float


class InputSection(Section):
    root_folder: Path | None
    project: str | None
    area: area_specs.AreaSpecification
    normal: NormalReceivers | None
    grid: GridReceivers | None

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        if 'project' in raw:
            self.project = str(raw['project'])
        else:
            self.project = None

        if 'root_folder' in raw:
            self.root_folder = Path(raw['root_folder'])
        else:
            self.root_folder = None

        self.area = parse_area(raw['area'])

        if 'normal' in raw['receivers']:
            self.normal = NormalReceivers(**raw['normal'])
        else:
            self.normal = None

        if 'grid' in raw['receivers']:
            self.grid = GridReceivers(**raw['grid'])
        else:
            self.grid = None


def parse_area(raw) -> area_specs.AreaSpecification:
    r = raw

    if isinstance(r, (int, float)) or isinstance(r, (list, tuple)):
        if not isinstance(r, (list, tuple)):
            r = (r, r)
        r = tuple(r)
        assert len(r) == 2
        return area_specs.Extent(r)

    elif isinstance(r, dict) and 'center' in r and 'extent' in r:
        c, e = r['center'], r['extent']
        if not isinstance(e, (list, tuple)):
            e = (e, e)
        c, e = tuple(c), tuple(e)
        assert len(c) == 2 and len(e) == 2

        if not (e[0] >= 0 and e[1] >= 0):
            raise ConfigError(f'Extents must be nonnegative')

        crs = parse_crs_name(r['crs']) if 'crs' in r else CRS.WGS84
        return area_specs.CenterExtent(c, e, crs)

    elif isinstance(r, dict) and 'botleft' in r and 'topright' in r:
        bl, tr = r['botleft'], r['topright']
        bl, tr = tuple(bl), tuple(tr)
        assert len(bl) == 2 and len(tr) == 2

        if not (bl[0] < tr[0] and bl[1] < tr[1]):
            raise ConfigError(f'Invalid area corners')

        crs = parse_crs_name(r['crs']) if 'crs' in r else CRS.WGS84
        return area_specs.Corners(bl, tr, crs)

    else:
        raise ConfigError('Invalid area specification')
