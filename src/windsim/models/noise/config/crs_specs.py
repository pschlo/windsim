from __future__ import annotations

from dataclasses import dataclass

import pyproj
import pyproj.aoi


class CrsSpecification:
    pass


class UTM(CrsSpecification):
    def __init__(self, datum: str, epsg_range: tuple[float,float]):
        self.datum = datum
        self.epsg_range = epsg_range

class UTM_ETRS(UTM):
    def __init__(self):
        super().__init__('ETRS89', (25800, 25900))

class UTM_WGS(UTM):
    def __init__(self):
        super().__init__('WGS84', (32600, 32800))

@dataclass
class LocalBest(CrsSpecification):
    pass

@dataclass
class Custom(CrsSpecification):
    crs: pyproj.CRS
