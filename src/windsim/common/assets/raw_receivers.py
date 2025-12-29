from __future__ import annotations
from typing import Any

from planner import Asset, Recipe, inject, DataAsset
import tomllib

from .setup import SetupAsset


class RawReceiversAsset(DataAsset[dict[str, Any]]):
    pass


class RawReceiversRecipe(Recipe):
    _makes = RawReceiversAsset

    setup: SetupAsset = inject()

    def make(self):
        return RawReceiversAsset(
            self.setup.d['receiver']
        )
