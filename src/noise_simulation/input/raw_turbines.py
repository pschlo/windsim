import logging
from typing import override, cast
import pandas as pd
import xarray as xr
import numpy as np
from contextlib import contextmanager

from planner import Asset, Recipe, DataAsset, inject
from .turbine_types import MyTurbineTypesAsset
from ._utils import _as_fixed_str
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class RawTurbinesAsset(DataAsset[xr.Dataset]):
    pass


class RawTurbinesRecipe(Recipe[RawTurbinesAsset]):
    _makes = RawTurbinesAsset

    config: ConfigAsset = inject()
    existing_turbines: assets.ExistingTurbinesJson = inject()
    scenarios: assets.ScenariosJson = inject()
    turbine_types: MyTurbineTypesAsset = inject()

    @override
    @contextmanager
    def make(self):
        """Turbines, untransformed and possibly without elevation."""
        log.debug("  Preparing turbines") 

        new_turbines = self.scenarios.d[0]["turbine_placements"]

        # combine existing sound sources and new turbines
        df = (
            pd.DataFrame(self.existing_turbines.d + new_turbines)
            .drop(columns=["x", "y"], errors='ignore')
            .rename(columns=dict(placement_id="turbine"))
            .set_index("turbine")
        )

        # convert to xarray
        ds = xr.Dataset.from_dataframe(df)
        ds = _as_fixed_str(ds, ['turbine', 'turbine_type', 'status'], errors='ignore')
        ds['position_lonlat'] = xr.concat([ds['longitude'], ds['latitude']], dim='spatial')
        ds = ds.assign_coords(dict(spatial=['x', 'y'])).drop_vars(['longitude', 'latitude'])

        # DEBUG: Discard turbines whose model does not appear in turbinetypes.
        # This triggers a Dask computation.
        if self.config.d.debug.discard_invalid_turbines:
            def _is_valid(turbines: xr.Dataset, turbinetypes: xr.Dataset):
                return cast(xr.DataArray, xr.apply_ufunc(
                    np.isin, turbines['turbine_type'], turbinetypes['model'],
                    input_core_dims=[['turbine'], ['model']],
                    output_core_dims=[['turbine']]
                ))
            is_valid = xr.map_blocks(_is_valid, ds, args=[self.turbine_types.d]).compute()
            if not is_valid.all():
                log.warning(f"Discarding {sum(~is_valid.values)} invalid turbines")
                ds = ds.sel(turbine=is_valid)

        # Create required variables if they don't exist yet
        if 'ground_level_m' not in ds:
            ds['ground_level_m'] = 'turbine', np.full(ds.sizes['turbine'], np.nan)

        try:
            yield RawTurbinesAsset(ds)
        finally:
            print("CLEANING UP RAW TURBINES")
