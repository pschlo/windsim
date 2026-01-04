import logging
from typing import override, cast, Any
import pandas as pd
import xarray as xr
import numpy as np

from planner import Asset, Recipe, DataAsset, inject

from windsim.common import assets

from ._utils import xr_as_dtype
from .frequencies import FrequenciesAsset


log = logging.getLogger(__name__)


class TurbineModelsAsset(DataAsset[xr.Dataset]):
    pass


class TurbineModelsRecipe(Recipe[TurbineModelsAsset]):
    _makes = TurbineModelsAsset

    raw_turbine_models: assets.TurbineModelsDict = inject()
    frequencies: FrequenciesAsset = inject()

    @override
    def make(self):
        # Load as dataframe
        df = (
            pd.DataFrame.from_dict(self.raw_turbine_models.d, orient="index")
            .rename_axis("model")
        )

        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df)

        # Type validation
        ds = xr_as_dtype(ds, dict(
            model="str",
            manufacturer=("str", "<MISSING>"),
            sound_power_db="object",
        ))

        # Validate length
        n = len(self.frequencies.d)
        lengths = xr.apply_ufunc(
            len,
            ds["sound_power_db"],
            vectorize=True,          # apply elementwise over model
            output_dtypes=[int],
        )
        bad = ds["model"].where(lengths != n, drop=True)
        if bad.size:
            bad_model = bad.values[0]
            raise ValueError(
                f"sound_power_db length mismatch for model '{bad_model}' (expected length: {n})"
            )

        # Overwrite sound power levels
        ds["sound_power_db"] = xr.DataArray(
            np.vstack(ds["sound_power_db"].values),  # type: ignore
            dims=("model", "frequency"),
            coords=dict(
                model=ds["model"],
                frequency=self.frequencies.d
            ),
        )

        # Type validation
        ds = xr_as_dtype(ds, dict(
            model="str",
            manufacturer="str",
            sound_power_db="float",
            frequency="int"
        ))

        return TurbineModelsAsset(ds)
