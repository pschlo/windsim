from collections.abc import Generator
from typing import override

from ....assets.lib import Asset, Recipe, inject, DataAsset
from .config import Config, ConfigData


class ConfigAsset(DataAsset[ConfigData]):
    pass
