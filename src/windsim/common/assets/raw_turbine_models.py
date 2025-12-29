from __future__ import annotations
from typing import Any

from planner import Asset, Recipe, inject, DataAsset
import tomllib

from .setup import SetupAsset


class RawTurbineModelsAsset(DataAsset[dict[str, Any]]):
    pass


class RawTurbineModelsRecipe(Recipe):
    _makes = RawTurbineModelsAsset

    setup: SetupAsset = inject()

    def make(self):
        return RawTurbineModelsAsset(
            self.setup.d['turbine_model']
        )
