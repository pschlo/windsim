from __future__ import annotations
from typing import Any, TypedDict, NotRequired, Literal

from planner import Asset, Recipe, inject, DataAsset
import tomllib

from .setup import SetupAsset


class TurbineDict(TypedDict):
    name: str
    model: str
    status: Literal["new", "operating"]
    position_lonlat: list[float]
    elevation_m: NotRequired[float]


class TurbinesDictAsset(DataAsset[list[TurbineDict]]):
    pass


class TurbinesDictRecipe(Recipe):
    _makes = TurbinesDictAsset

    setup: SetupAsset = inject()

    def make(self):
        return TurbinesDictAsset(
            self.setup.d['turbine']
        )
