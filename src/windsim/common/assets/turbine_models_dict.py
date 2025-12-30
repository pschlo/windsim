from __future__ import annotations
from typing import Any, TypedDict, NotRequired

from planner import Asset, Recipe, inject, DataAsset
import tomllib

from .setup import SetupAsset


class TurbineModel(TypedDict):
    sound_power_db: list[float]
    manufacturer: str


class TurbineModelsDictAsset(DataAsset[dict[str, TurbineModel]]):
    pass


class TurbineModelsDictRecipe(Recipe):
    _makes = TurbineModelsDictAsset

    setup: SetupAsset = inject()

    def make(self):
        return TurbineModelsDictAsset(
            self.setup.d['turbine_model']
        )
