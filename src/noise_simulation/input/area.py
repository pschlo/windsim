import logging
import pyproj
from typing import override
from dataclasses import dataclass

from planner import Asset, Recipe, DataAsset, inject, assets
from ..config import area_specs, ConfigAsset
from .area_of_interest import AreaOfInterestAsset
from .working_crs import WorkingCrsAsset
from .transformed_turbines import TransformedTurbinesAsset


log = logging.getLogger(__name__)


Point = tuple[int|float, int|float]

@dataclass
class Area:
    botleft: Point
    topright: Point


class AreaAsset(DataAsset[Area]):
    pass


class AreaRecipe(Recipe[AreaAsset]):
    _makes = AreaAsset

    config: ConfigAsset = inject()
    aoi: AreaOfInterestAsset = inject()
    crs: WorkingCrsAsset = inject()
    turbines: TransformedTurbinesAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing area")

        specs = self.config.d.input.area

        match specs:
            case area_specs.Extent():
                xmin, ymin = self.turbines.d['position'].min('turbine').sel(spatial=['x', 'y']).values
                xmax, ymax = self.turbines.d['position'].max('turbine').sel(spatial=['x', 'y']).values
                botleft = (xmin - specs.extent[0], ymin - specs.extent[1])
                topright = (xmax + specs.extent[0], ymax + specs.extent[1])
                return AreaAsset(Area(botleft, topright))
            case area_specs.CenterExtent():
                t = pyproj.Transformer.from_crs(specs.crs, self.crs.d, always_xy=True, area_of_interest=self.aoi.d)
                center = t.transform(*specs.center)
                return AreaAsset(Area(
                    (center[0] - specs.extent[0]/2, center[1] - specs.extent[1]/2),
                    (center[0] + specs.extent[0]/2, center[1] + specs.extent[1]/2)
                ))
            case area_specs.Corners():
                t = pyproj.Transformer.from_crs(specs.crs, self.crs.d, always_xy=True, area_of_interest=self.aoi.d)
                xmin, ymin, xmax, ymax = t.transform_bounds(*specs.botleft, *specs.topright)
                return AreaAsset(Area((xmin, ymin), (xmax, ymax)))
            case _:
                assert False
