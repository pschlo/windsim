from typing import override
from pathlib import Path
from dataclasses import dataclass

from planner import Asset, Recipe, inject, DataAsset, store
from .config import Config, ConfigData


class ConfigAsset(DataAsset[ConfigData]):
    pass


class ConfigRecipe(Recipe[ConfigAsset]):
    _makes = ConfigAsset
    _caps = [
        store.StorageCap(tag=".", shared=True)
    ]

    storage: store.assets.StorageProvider = inject()

    def make(self):
        path = self.storage.persistent_dir() / "config.toml"
        config = Config.load(path)
        return ConfigAsset(config.data)
