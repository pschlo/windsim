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


INPUT_ATTRS = {
    'sound_power_db',
    'manufacturer'
}


# Mapping from dataset variable name to dtype and optional placeholder for NaN-like values
DTYPES: dict = dict(
    model="str",
    sound_power_db="float",
    manufacturer=("str", "<MISSING>"),
    frequency="int",
)


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

        # Validate columns
        if extra_vars := set(df.columns) - INPUT_ATTRS:
            raise ValueError(f"Unexpected attribute '{next(iter(extra_vars))}'")

        df = df.reindex(columns=list(INPUT_ATTRS))

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
        ds = xr.Dataset.from_dataframe(
            df
            .drop(columns=["sound_power_db"])
        )

        # Add sound power levels
        ds["sound_power_db"] = xr.DataArray(
            np.vstack(df["sound_power_db"]),  # type: ignore
            dims=("model", "frequency"),
            coords=dict(
                model=df.index,
                frequency=self.frequencies.d
            ),
        )

        # Xarray assertions
        for name, var in ds.variables.items():
            a = DTYPES[name]
            if not isinstance(a, tuple | list):
                a = (a, None)
            elif len(a) == 1:
                a = (a[0], None)
            else:
                assert len(a) == 2
            dtype, placeholder = a

            # Fill placeholder
            if placeholder is not None:
                var = var.fillna(placeholder)
            else:
                if var.isnull().any():
                    raise ValueError(f"{name} missing for some models")

            # Convert type
            ds[name] = var.astype(dtype, copy=False)
        assert set(ds.variables) == set(DTYPES)

        return TurbineModelsAsset(ds)
