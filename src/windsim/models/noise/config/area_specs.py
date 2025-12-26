from __future__ import annotations

from dataclasses import dataclass

import pyproj
import pyproj.aoi

Point = tuple[int|float, int|float]

class AreaSpecification:
    pass

@dataclass
class CenterExtent(AreaSpecification):
    center: Point
    extent: Point
    crs: pyproj.CRS

@dataclass
class Extent(AreaSpecification):
    extent: Point

@dataclass
class Corners(AreaSpecification):
    botleft: Point
    topright: Point
    crs: pyproj.CRS
