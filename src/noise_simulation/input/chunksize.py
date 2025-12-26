import logging
from typing import override
import os

from planner import Asset, Recipe, DataAsset, inject, assets
from ..config import chunksize_specs, ConfigAsset
from .area import AreaAsset


log = logging.getLogger(__name__)


class Chunksize:
    def __init__(self, size: int) -> None:
        self._1d = size
        self._2d = int(size**(1/2)) if size >= 0 else -1
        self._3d = int(size**(1/3)) if size >= 0 else -1


class ChunksizeAsset(DataAsset[Chunksize]):
    pass


class ChunksizeRecipe(Recipe[ChunksizeAsset]):
    _makes = ChunksizeAsset

    config: ConfigAsset = inject()
    area: AreaAsset = inject()

    @override
    def make(self):
        area = self.area.d
        specs = self.config.d.computation.chunksize

        log.debug("  Preparing chunksize")

        match specs:
            case chunksize_specs.Custom():
                return ChunksizeAsset(
                    Chunksize(specs.value)
                )
            case chunksize_specs.Disabled():
                return ChunksizeAsset(
                    Chunksize(-1)
                )
            case chunksize_specs.Auto() | chunksize_specs.CappedAuto():
                if self.config.d.input.grid is None:
                    return ChunksizeAsset(
                        Chunksize(-1)
                    )  # just disable chunking for now.
                xmin, ymin = area.botleft
                xmax, ymax = area.topright
                num_x = int((xmax - xmin) / self.config.d.input.grid.spacing) + 1
                num_y = int((ymax - ymin) / self.config.d.input.grid.spacing) + 1
                num_cores = os.cpu_count()
                assert num_cores is not None

                chunksize = (num_x * num_y) // (3 * num_cores)
                if isinstance(specs, chunksize_specs.CappedAuto):
                    return ChunksizeAsset(
                        Chunksize(min(chunksize, specs.max))
                    )
                else:
                    return ChunksizeAsset(
                        Chunksize(chunksize)
                    )
            case _:
                assert False
