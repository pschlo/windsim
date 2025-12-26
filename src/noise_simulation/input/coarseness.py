import logging
from typing import override
import xarray as xr

from planner import Asset, Recipe, DataAsset, inject, assets
from .full_turbines import FullTurbinesAsset
from .receivers import ReceiversAsset
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class CoarsenessAsset(DataAsset[xr.Dataset]):
    pass


class CoarsenessRecipe(Recipe[CoarsenessAsset]):
    _makes = CoarsenessAsset
    
    config: ConfigAsset = inject()
    turbines: FullTurbinesAsset = inject()
    receivers: ReceiversAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing coarseness")

        # coarseness factors between 0 and 1
        # TODO: Fill with actual data.
        coarseness = xr.Dataset(
            dict(
                G_m=0.8,
                G_r=0.8,
                G_s=0.8
            )
        )

        return CoarsenessAsset(coarseness)
