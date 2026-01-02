import logging
from typing import override, cast
import pandas as pd
import xarray as xr
import numpy as np
from contextlib import contextmanager

from planner import Asset, Recipe, DataAsset, inject

from windsim.common import assets

from .turbine_models import TurbineModelsAsset
from ._utils import _as_fixed_str
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class BaseTurbinesAsset(DataAsset[xr.Dataset]):
    pass


class BaseTurbinesRecipe(Recipe[BaseTurbinesAsset]):
    _makes = BaseTurbinesAsset

    config: ConfigAsset = inject()
    turbines: assets.TurbinesDict = inject()
    turbine_models: TurbineModelsAsset = inject()

    @override
    def make(self):
        """Turbines, untransformed and possibly without elevation."""
        log.debug("  Preparing turbines")

        # combine existing sound sources and new turbines
        df = (
            pd.DataFrame(self.turbines.d)
            .rename(columns=dict(name="turbine"))
            .set_index("turbine")
        )

        # convert to xarray
        ds = xr.Dataset.from_dataframe(df)
        ds = _as_fixed_str(ds, ['turbine', 'model', 'status'], errors='ignore')

        ds["position_lonlat"] = xr.DataArray(
            data=np.stack(ds["position_lonlat"].values),  # type: ignore
            dims=("turbine", "spatial"),
            coords=dict(
                turbine=ds.coords["turbine"],
                spatial=["x", "y"]
            ),
        )

        # DEBUG: Discard turbines whose model does not appear in turbinetypes.
        # This triggers a Dask computation.
        if self.config.d.debug.discard_invalid_turbines:
            def _is_valid(turbines: xr.Dataset, turbinetypes: xr.Dataset):
                return cast(xr.DataArray, xr.apply_ufunc(
                    np.isin, turbines['model'], turbinetypes['model'],
                    input_core_dims=[['turbine'], ['model']],
                    output_core_dims=[['turbine']]
                ))
            is_valid = xr.map_blocks(_is_valid, ds, args=[self.turbine_models.d]).compute()
            if not is_valid.all():
                log.warning(f"Discarding {sum(~is_valid.values)} invalid turbines")
                ds = ds.sel(turbine=is_valid)

        # Create required variables if they don't exist yet
        if 'elevation_m' not in ds:
            ds['elevation_m'] = 'turbine', np.full(ds.sizes['turbine'], np.nan)

        return BaseTurbinesAsset(ds)
