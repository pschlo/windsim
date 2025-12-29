import xarray as xr
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import override, Any, cast

from planner import Asset, DataAsset, Recipe, inject

from windsim.models.noise import assets as noise_assets


class AdaptedTurbinesAsset(DataAsset[xr.Dataset]):
    pass


class AdaptedTurbinesRecipe(Recipe[AdaptedTurbinesAsset]):
    _makes = AdaptedTurbinesAsset

    turbines: noise_assets.FullTurbines = inject()
    turbine_types: noise_assets.TurbineModels = inject()

    @override
    def make(self):
        # Match turbine types with turbines
        _ds = (
            self.turbine_types.d[['rotor_diameter_m', 'max_blade_depth_m', 'blade_depth_90_percent_m']]
            .sel(model=self.turbines.d['turbine_type'])
            .drop_vars('model')
        )

        ds = self.turbines.d.copy()
        # Average chord length of a rotor blade
        ds['c_bar'] = 0.5 * (_ds['max_blade_depth_m'] + _ds['blade_depth_90_percent_m'])
        ds['rotor_radius_m'] = _ds['rotor_diameter_m'] / 2

        delta = np.deg2rad(0.53)   # ~0.0092 rad
        ds['L_max'] = ds['c_bar'] / (0.2 * delta)  # (turbine,)

        return AdaptedTurbinesAsset(ds)
