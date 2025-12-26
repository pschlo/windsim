import logging
from typing import override, cast
import pandas as pd
import xarray as xr
import numpy as np

from planner import Asset, Recipe, DataAsset, inject, assets
from ._utils import _as_fixed_str
from .frequencies import FrequenciesAsset


log = logging.getLogger(__name__)


class MyTurbineTypesAsset(DataAsset[xr.Dataset]):
    pass


class MyTurbineTypesRecipe(Recipe[MyTurbineTypesAsset]):
    _makes = MyTurbineTypesAsset

    turbine_types: assets.TurbineTypesJson = inject()
    frequencies: FrequenciesAsset = inject()

    @override
    def make(self):
        df = pd.DataFrame(self.turbine_types.d)

        # Raise if octave_frequencies_db is None for any model
        bad_none = df['octave_frequencies_db'].isna()
        if bad_none.any():
            model = df.loc[bad_none, 'model'].iloc[0]
            raise ValueError(f"octave_frequencies_db is None for model '{model}'")

        # convert to xarray
        ds = xr.Dataset.from_dataframe(
            df
            .drop(columns=['octave_frequencies_db'])
            .set_index("model")
        )

        # Add sound power levels
        ds['sound_power_level_db'] = xr.DataArray.from_series(
            df
            [['model', 'octave_frequencies_db']]
            .assign(frequency=lambda df: [self.frequencies.d] * len(df))
            .explode(['frequency', 'octave_frequencies_db'])
            .set_index(['model', 'frequency'])
            ['octave_frequencies_db']
            .astype(np.float64)
        )

        ds = _as_fixed_str(ds, ['model', 'manufacturer'])
        return MyTurbineTypesAsset(ds)
