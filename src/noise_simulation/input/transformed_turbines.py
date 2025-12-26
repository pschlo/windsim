import logging
import pyproj
from typing import override, cast
import xarray as xr
import numpy as np
from contextlib import contextmanager

from planner import Asset, Recipe, DataAsset, inject
from noise_simulation.coordinate_reference_systems import CRS

from .raw_turbines import RawTurbinesAsset
from .working_crs import WorkingCrsAsset
from .area_of_interest import AreaOfInterestAsset
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class TransformedTurbinesAsset(DataAsset[xr.Dataset]):
    pass


class TransformedTurbinesRecipe(Recipe[TransformedTurbinesAsset]):
    _makes = TransformedTurbinesAsset

    config: ConfigAsset = inject()
    turbines: RawTurbinesAsset = inject()
    crs: WorkingCrsAsset = inject()
    aoi: AreaOfInterestAsset = inject()

    @override
    @contextmanager
    def make(self):
        """Transform turbines to working CRS."""
        log.debug("  Preparing transformed turbines")

        t = pyproj.Transformer.from_crs(CRS.WGS84, self.crs.d, always_xy=True, area_of_interest=self.aoi.d)
        x, y = t.transform(*self.turbines.d['position_lonlat'].transpose('spatial', ...).sel(spatial=['x', 'y']).values)
        ds = self.turbines.d.assign(position=(('spatial', 'turbine'), [x, y]))

        # DEBUG: Offset turbine positions by random direction and distance.
        _jitter = self.config.d.debug.jitter_turbine_positions
        if _jitter is not None:
            max_dist = _jitter.max_offset_distance
            z_score = 2  # 95.4 % interval
            std_dev = max_dist / z_score
            # generate random distances in interval [0, max_dist]
            distance = np.abs(
                self.config.d.rng.normal(0, std_dev, ds.sizes['turbine'])
            ).clip(0, max_dist)
            angle = self.config.d.rng.uniform(0, 2*np.pi, ds.sizes['turbine'])
            offset_xy = distance[:, np.newaxis] * np.column_stack([np.sin(angle), np.cos(angle)])
            offset = xr.DataArray(offset_xy, dims=['turbine', 'spatial'])
            ds['position'] += offset

        try:
            yield TransformedTurbinesAsset(ds)
        finally:
            print("CLEANING UP TRANSFORMED TURBINES")
