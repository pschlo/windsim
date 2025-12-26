import numpy as np
import logging
from typing import override

from planner import Asset, Recipe, DataAsset, inject, assets
from noise_simulation.coordinate_reference_systems import CRS


log = logging.getLogger(__name__)


class FrequenciesAsset(DataAsset[np.ndarray]):
    pass


class FrequenciesRecipe(Recipe[FrequenciesAsset]):
    _makes = FrequenciesAsset

    @override
    def make(self):
        return FrequenciesAsset(
            np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        )
