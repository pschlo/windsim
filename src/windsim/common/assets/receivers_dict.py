from __future__ import annotations
from typing import Any, TypedDict, NotRequired

from planner import Asset, Recipe, inject, DataAsset
import tomllib

from .setup import SetupAsset


class ReceiverDict(TypedDict):
    name: str
    position_lonlat: list[float]
    elevation_m: NotRequired[float]


class ReceiversDictAsset(DataAsset[list[ReceiverDict]]):
    pass


class ReceiversDictRecipe(Recipe):
    _makes = ReceiversDictAsset

    setup: SetupAsset = inject()

    def make(self):
        return ReceiversDictAsset(
            self.setup.d['receiver']
        )
