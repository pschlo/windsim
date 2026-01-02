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


COLUMNS = {
    'sound_power_db',
    'manufacturer'
}


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

        # Validate column exists
        if missing_vals := COLUMNS - set(df.columns):
            raise ValueError(f"{next(iter(missing_vals))} is missing for all models")

        # Validate not NaN
        bad_none = df['sound_power_db'].isna()
        if bad_none.any():
            model = df.index[bad_none][0]
            raise ValueError(f"sound_power_db is missing for model '{model}'")

        # Validate length
        n = len(self.frequencies.d)
        bad_length = df["sound_power_db"].map(len) != n
        if bad_length.any():
            bad = df.index[bad_length][0]
            raise ValueError(f"sound_power_db length mismatch for model '{bad}' (expected length: {n})")

        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df.drop(columns=["sound_power_db"]))

        # Add sound power levels
        ds["sound_power_db"] = xr.DataArray(
            np.vstack(df["sound_power_db"].to_list()).astype(float),
            coords={"model": df.index.to_numpy(), "frequency": self.frequencies.d},
            dims=("model", "frequency"),
        )

        ds = _as_fixed_str(ds, ['model', 'manufacturer'])
        return TurbineModelsAsset(ds)
