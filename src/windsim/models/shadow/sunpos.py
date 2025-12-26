import xarray as xr
import numpy as np
import pandas as pd
import dask.array.creation as da_creation
import dask.array.core as da_core
from dataclasses import dataclass
from typing import override, Any, cast
from soltrack import SolTrack
import pyproj

from planner import Asset, DataAsset, Recipe, inject

from windsim.coordinate_reference_systems import CRS
from windsim.models.noise import assets as noise_assets


@dataclass
class SunposConfAsset(Asset):
    frequency: str = "1min"
    year: int = 2025


class SunposAsset(DataAsset[xr.Dataset]):
    pass


class SunposRecipe(Recipe[SunposAsset]):
    _makes = SunposAsset

    conf: SunposConfAsset = inject()
    area: noise_assets.Area = inject()
    working_crs: noise_assets.WorkingCrs = inject()
    aoi: noise_assets.AreaOfInterest = inject()

    @override
    def make(self):
        ds = xr.Dataset()

        # Build time axis: tz-naive UTC, tell SolTrack utc=True
        ds = ds.assign_coords(
            time=pd.date_range(f"{self.conf.year}-01-01", f"{self.conf.year+1}-01-01", freq=self.conf.frequency, inclusive="left")
        )

        a = self.area.d
        center = (a.botleft[0] + a.topright[0]) / 2, (a.botleft[1] + a.topright[1]) / 2

        t = pyproj.Transformer.from_crs(self.working_crs.d, CRS.WGS84, always_xy=True, area_of_interest=self.aoi.d)
        lon, lat = t.transform(*center)

        _azimuth, _altitude = _solpos_one_point(lon, lat, ds['time'])
        ds['azimuth_rad'] = ('time', _azimuth)
        ds['altitude_rad'] = ('time', _altitude)

        ds['v_sun'] = xr.concat(
            cast(
                list[xr.DataArray],
                [
                    np.cos(ds['altitude_rad']) * np.sin(ds['azimuth_rad']),
                    np.cos(ds['altitude_rad']) * np.cos(ds['azimuth_rad']),
                    np.sin(ds['altitude_rad'])
                ]
            ),
            dim='spatial'
        ).assign_coords(spatial=['x', 'y', 'z']).astype(np.float32, copy=False)

        return SunposAsset(ds)


def _solpos_one_point(lon, lat, time_np64):
    """
    SolTrack wrapper for one location and a whole time vector.
    Inputs:
      lon, lat: floats (degrees, lon east-positive)
      time_np64: 1D np.datetime64[ns] array (UTC, tz-naive)
    Returns:
      (azimuth[time], altitude[time], distance[time])
    """
    # Build a tz-naive pandas index and tell SolTrack it's UTC (fast path)
    t = pd.DatetimeIndex(time_np64)
    # print(f"Time: {t}")
    st = SolTrack(np.deg2rad(lon), np.deg2rad(lat), use_degrees=False, use_north_equals_zero=True, compute_distance=False)
    st.set_date_time(t, utc=True)
    st.compute_position()
    return st.azimuth, st.altitude
