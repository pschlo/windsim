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

    raw_turbine_models: assets.RawTurbineModels = inject()
    frequencies: FrequenciesAsset = inject()

    @override
    def make(self):
        df = pd.DataFrame(self.raw_turbine_models.d)

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
        return TurbineModelsAsset(ds)
