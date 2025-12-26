import pyproj
import pyproj.database
import logging
from typing import override, cast

from planner import Asset, Recipe, DataAsset, inject
from ..config import ConfigAsset, crs_specs
from .area_of_interest import AreaOfInterestAsset


log = logging.getLogger(__name__)


class WorkingCrsAsset(DataAsset[pyproj.CRS]):
    pass


class WorkingCrsRecipe(Recipe[WorkingCrsAsset]):
    _makes = WorkingCrsAsset

    config: ConfigAsset = inject()
    aoi: AreaOfInterestAsset = inject()

    @override
    def make(self):
        """Determines the Coordinate Reference System for all calculations throughout the simulation.
        Makes use of area information and queries the pyproj database.
        """
        log.debug("  Preparing working CRS")

        aoi = self.aoi.d
        specs = self.config.d.computation.crs

        match specs:
            case crs_specs.UTM():
                infos = pyproj.database.query_utm_crs_info(area_of_interest=aoi, datum_name=specs.datum, contains=True)
                if not infos:
                    infos = pyproj.database.query_utm_crs_info(area_of_interest=aoi, datum_name=specs.datum, contains=False)
                infos = [x for x in infos if x.auth_name == 'EPSG' and specs.epsg_range[0] <= int(x.code) < specs.epsg_range[1]]
                crs = pyproj.CRS.from_epsg(infos[0].code)
            case crs_specs.LocalBest():
                center_lon = aoi.west_lon_degree + (aoi.east_lon_degree - aoi.west_lon_degree)/2
                center_lat = aoi.south_lat_degree + (aoi.north_lat_degree - aoi.south_lat_degree)/2
                # Create custom Transverse Mercator CRS centered at area center
                crs = pyproj.CRS(proj="tmerc", lat_0=center_lat, lon_0=center_lon, datum="WGS84", units="m")
            case crs_specs.Custom():
                crs = specs.crs
            case _:
                assert False

        return WorkingCrsAsset(crs)
