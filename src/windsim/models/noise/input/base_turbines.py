import logging
from typing import override, cast, Any
import pandas as pd
import xarray as xr
import numpy as np
from contextlib import contextmanager

from planner import Asset, Recipe, DataAsset, inject

from windsim.common import assets

from .turbine_models import TurbineModelsAsset
from ._utils import xr_as_dtype
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

        df = pd.DataFrame(self.turbines.d)

        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df)
        ds = xr_as_dtype(ds, dict(
            index="int",
            name=("str", "<unnamed>"),
            model="str",
            status=("str", "new"),
            hub_height_m="float",
            position_lonlat="object",
            elevation_m=("float", np.nan)
        ))

        # Create index
        ds["name"] = ds["name"].broadcast_like(ds["index"])
        ds = (
            ds
            .rename(name="turbine")
            .set_coords("turbine")
            .swap_dims(index="turbine")
            .drop_vars("index")
        )

        ds["position_lonlat"] = xr.DataArray(
            data=np.stack(ds["position_lonlat"].values),  # type: ignore
            dims=("turbine", "spatial"),
            coords=dict(
                turbine=ds["turbine"],
                spatial=["x", "y"]
            ),
        )

        ds = xr_as_dtype(ds, dict(
            turbine="str",
            model="str",
            status="str",
            spatial="str",
            hub_height_m="float",
            position_lonlat="float",
            elevation_m=("float", np.nan)
        ))


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

        return BaseTurbinesAsset(ds)
