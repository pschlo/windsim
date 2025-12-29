from __future__ import annotations
from typing import Any

from planner import Asset, Recipe, inject, DataAsset
import tomllib

from .setup import SetupAsset


class RawTurbinesAsset(DataAsset[dict[str, Any]]):
    pass


class RawTurbinesRecipe(Recipe):
    _makes = RawTurbinesAsset

    setup: SetupAsset = inject()

    def make(self):
        return RawTurbinesAsset(
            self.setup.d['turbine']
        )
