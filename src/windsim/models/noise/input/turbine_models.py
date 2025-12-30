import logging
from typing import override, cast
import pandas as pd
import xarray as xr
import numpy as np

from planner import Asset, Recipe, DataAsset, inject

from windsim.common import assets

from ._utils import _as_fixed_str
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
        df = (
            pd.DataFrame.from_dict(self.raw_turbine_models.d, orient="index")
            .rename_axis("model")
        )

        # Validate lengths
        n = len(self.frequencies.d)
        lengths = df["sound_power_db"].map(len)
        if not (lengths == n).all():
            bad = df.index[lengths != n].to_list()
            raise ValueError(f"sound_power_db length mismatch for models {bad} (expected length: {n})")

        # Explode sound power levels
        spl = xr.DataArray(
            np.vstack(df["sound_power_db"].to_list()).astype(float),
            coords={"model": df.index.to_numpy(), "frequency": self.frequencies.d},
            dims=("model", "frequency"),
            name="sound_power_db",
        )

        # everything else becomes dataset vars (one value per model)
        ds = xr.Dataset.from_dataframe(df.drop(columns=["sound_power_db"]))

        # attach the 2D variable
        ds["sound_power_db"] = spl

        ds = _as_fixed_str(ds, ['model', 'manufacturer'])
        return TurbineModelsAsset(ds)
