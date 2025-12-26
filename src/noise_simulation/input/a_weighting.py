import xarray as xr
import logging
from typing import override

from planner import Asset, Recipe, inject, DataAsset


log = logging.getLogger(__name__)


class AWeightingAsset(DataAsset[xr.DataArray]):
    pass


class AWeightingRecipe(Recipe[AWeightingAsset]):
    _makes = AWeightingAsset

    @override
    def make(self):
        data = xr.DataArray(
            [-26.2, -16.1, -8.6, -3.2, 0, +1.2, +1.0, -1.1],
            coords={
                'frequency': [63, 125, 250, 500, 1000, 2000, 4000, 8000]
            }
        )
        return AWeightingAsset(data=data)
