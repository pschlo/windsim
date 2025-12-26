import logging
from typing import override
import pandas as pd
import xarray as xr
import numpy as np

from planner import Asset, Recipe, DataAsset, inject
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class AtmosphericCoefficientAsset(DataAsset[xr.DataArray]):
    pass


class AtmosphericCoefficientRecipe(Recipe[AtmosphericCoefficientAsset]):
    _makes = AtmosphericCoefficientAsset
    
    config: ConfigAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing atmospheric coefficient")
        alpha_series = (
            pd.DataFrame(
                [
                    [10, 70, [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117]],
                    [20, 70, [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]],
                    [30, 70, [0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3]],
                    [15, 20, [0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202]],
                    [15, 50, [0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129]],
                    [15, 80, [0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8]]
                ],
                columns=['temperature', 'humidity', 'alpha']
            )
            .assign(frequency=lambda df: [[63, 125, 250, 500, 1000, 2000, 4000, 8000]] * len(df))
            .explode(['frequency', 'alpha'])
            .set_index(['temperature', 'humidity', 'frequency'])
            ['alpha']
            .astype(np.float64)
        )
        atmospheric_coefficient = xr.DataArray.from_series(alpha_series)

        # DEBUG
        atmospheric_coefficient = atmospheric_coefficient.sel(humidity=70, temperature=10)

        return AtmosphericCoefficientAsset(atmospheric_coefficient)
