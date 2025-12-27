from __future__ import annotations
from typing import override, Any, Unpack
from xarray import DataArray
from dataclasses import dataclass
from typing import TypedDict
import pyproj
from pathlib import Path

from planner import Asset, Recipe, inject, DataAsset

from windsim.common.data_sources.fabdem import get_elevation


class FabdemAsset(DataAsset[DataArray]):
    pass


@dataclass
class FabdemConfAsset(Asset):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    crs: pyproj.CRS
    exact_bounds: bool = True
    chunksize: int | None = None


class FabdemRecipe(Recipe[FabdemAsset]):
    _makes = FabdemAsset
    _dir = "fabdem"
    _shared = True

    config: FabdemConfAsset = inject()

    @override
    def make(self):
        elevation = get_elevation(
            xmin=self.config.xmin,
            ymin=self.config.ymin,
            xmax=self.config.xmax,
            ymax=self.config.ymax,
            crs=self.config.crs,
            folder=self.workdir,
            exact_bounds=self.config.exact_bounds
        )
        elevation.chunk(self.config.chunksize)
        return FabdemAsset(elevation)
