from __future__ import annotations
from typing import Any

from planner import Asset, Recipe, inject, DataAsset
import tomllib


class SetupAsset(DataAsset[dict[str, Any]]):
    pass


class SetupRecipe(Recipe[SetupAsset]):
    _makes = SetupAsset
    _dir = ""

    def make(self):
        with open(self.workdir / "setup.toml", 'rb') as f:
            data = tomllib.load(f)
        return SetupAsset(data)
