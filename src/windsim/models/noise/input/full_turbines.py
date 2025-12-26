import logging
from typing import override
import xarray as xr
from contextlib import contextmanager

from planner import Asset, Recipe, DataAsset, inject
from .transformed_turbines import TransformedTurbinesAsset
from .chunksize import ChunksizeAsset
from .elevation import ElevationAsset
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class FullTurbinesAsset(DataAsset[xr.Dataset]):
    pass


class FullTurbinesRecipe(Recipe[FullTurbinesAsset]):
    _makes = FullTurbinesAsset

    config: ConfigAsset = inject()
    turbines: TransformedTurbinesAsset = inject()
    elevation: ElevationAsset = inject()
    chunksize: ChunksizeAsset = inject()

    @override
    @contextmanager
    def make(self):
        """Create fully prepared turbines that include elevation data."""
        log.debug("  Preparing turbines")

        ds = self.turbines.d.chunk(turbine=self.chunksize.d._1d)

        # use real elevation data if elevation not specified
        real_elevations = (
            self.elevation.d
            .interp(
                x=ds['position'].sel(spatial='x'),
                y=ds['position'].sel(spatial='y')
            )
            .drop_vars(['x', 'y', 'spatial_ref'], errors='ignore')
        )
        ds['ground_level_m'] = ds['ground_level_m'].fillna(real_elevations)

        # add Z coordinate to position
        position_z = ds['ground_level_m'] + ds['hub_height_m']
        ds = ds.reindex(spatial=['x', 'y', 'z']).chunk(spatial=-1)
        ds['position'].loc[dict(spatial='z')] = position_z

        try:
            yield FullTurbinesAsset(ds)
        finally:
            print("CLEANING UP FULL TURBINES")
