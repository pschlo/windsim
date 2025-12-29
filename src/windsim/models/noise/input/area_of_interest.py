import pyproj.aoi
import logging
from typing import override
from dataclasses import dataclass

from planner import Asset, Recipe, inject, DataAsset

from windsim.coordinate_reference_systems import CRS

from .base_turbines import BaseTurbinesAsset
from ..config import area_specs, ConfigAsset


log = logging.getLogger(__name__)


class AreaOfInterestAsset(DataAsset[pyproj.aoi.AreaOfInterest]):
    pass


class AreaOfInterestRecipe(Recipe[AreaOfInterestAsset]):
    _makes = AreaOfInterestAsset

    turbines: BaseTurbinesAsset = inject()
    config: ConfigAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing area of interest")
        
        specs = self.config.data.input.area
        match specs:
            case area_specs.Corners():
                t = pyproj.Transformer.from_crs(specs.crs, CRS.WGS84, always_xy=True)
                bl_wgs, tr_wgs = t.transform(*specs.botleft), t.transform(*specs.topright)
                return AreaOfInterestAsset(
                    data=pyproj.aoi.AreaOfInterest(bl_wgs[0], bl_wgs[1], tr_wgs[0],tr_wgs[1])
                )
            case area_specs.CenterExtent():
                t = pyproj.Transformer.from_crs(specs.crs, CRS.WGS84, always_xy=True)
                center_wgs = t.transform(*specs.center)
                size = 0.000005  # dummy size
                return AreaOfInterestAsset(
                    data=pyproj.aoi.AreaOfInterest(center_wgs[0]-size, center_wgs[1]-size, center_wgs[0]+size, center_wgs[1]+size)
                )
            case area_specs.Extent():
                xmin, ymin = self.turbines.data['position_lonlat'].min('turbine').sel(spatial=['x', 'y']).values
                xmax, ymax = self.turbines.data['position_lonlat'].max('turbine').sel(spatial=['x', 'y']).values
                return AreaOfInterestAsset(
                    data=pyproj.aoi.AreaOfInterest(xmin, ymin, xmax, ymax)
                )
            case _:
                assert False
