from __future__ import annotations
from typing import override, TypedDict, Unpack, Any
from dataclasses import dataclass, field
from collections.abc import Collection
from xarray import Dataset
import xarray as xr
from pathlib import Path

from planner import Asset, Recipe, inject, DataAsset

from .cerra_store import CerraStoreAsset


class CerraHeightsAsset(DataAsset[Dataset]):
    pass

@dataclass
class CerraHeightsConfAsset(Asset):
    year: int | Collection[int] | None = None
    month: int | Collection[int] | None = None


class CerraHeightsRecipe(Recipe[CerraHeightsAsset]):
    _makes = CerraHeightsAsset

    store: CerraStoreAsset = inject()
    config: CerraHeightsConfAsset = inject()

    @override
    def make(self):
        ds = self.store.load(year=self.config.year, month=self.config.month)
        # TODO: filter down dataset
        return CerraHeightsAsset(ds)
