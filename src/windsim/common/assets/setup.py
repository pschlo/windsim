from __future__ import annotations
from typing import Any

from planner import Asset, Recipe, inject, DataAsset, store
import tomllib


class SetupAsset(DataAsset[dict[str, Any]]):
    pass


class SetupRecipe(Recipe[SetupAsset]):
    _makes = SetupAsset
    _caps = [
        store.StorageCap(tag=".")
    ]

    storage: store.assets.StorageProvider = inject()

    def make(self):
        with open(self.storage.get_persistent() / "setup.toml", 'rb') as f:
            data = tomllib.load(f)
        return SetupAsset(data)
