from collections.abc import Generator
import logging
from typing import override
import xarray as xr
from pathlib import Path

from planner import Asset, Recipe, inject, DataAsset
from .area import AreaAsset
from .chunksize import ChunksizeAsset
from .working_crs import WorkingCrsAsset
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class FabdemConfRecipe(Recipe[assets.FabdemConf]):
    _makes = assets.FabdemConf

    area: AreaAsset = inject()
    crs: WorkingCrsAsset = inject()
    chunksize: ChunksizeAsset = inject()

    @override
    def make(self):
        xmin, ymin = self.area.d.botleft
        xmax, ymax = self.area.d.topright
        return assets.FabdemConf(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            crs=self.crs.d,
            chunksize=self.chunksize.d._2d
        )


class ElevationAsset(DataAsset[xr.DataArray]):
    pass


class ElevationRecipe(Recipe[ElevationAsset]):
    _makes = ElevationAsset

    fabdem: assets.Fabdem = inject()
    config: ConfigAsset = inject()

    @override
    def make(self):
        # log.debug("  Preparing elevation data")
        return ElevationAsset(self.fabdem.d)
