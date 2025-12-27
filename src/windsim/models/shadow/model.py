import logging
import xarray as xr
import numpy as np
import pandas as pd
from typing import override, Any, cast
from dataclasses import dataclass

from planner import Asset, Recipe, DataAsset, inject

from windsim.common import assets as common_assets
from windsim.models.noise import assets as _na
from windsim.models.noise.config import ConfigAsset

from .sunpos import SunposConfAsset, SunposAsset
from .adapted_turbines import AdaptedTurbinesAsset


log = logging.getLogger(__name__)


@dataclass
class ShadowSimulationAsset(Asset):
    result: xr.Dataset
    receiver_groups: dict[str, xr.Dataset]


class ShadowSimulationRecipe(Recipe[ShadowSimulationAsset]):
    _makes = ShadowSimulationAsset

    turbine_types: _na.MyTurbineTypes = inject()
    area: _na.Area = inject()
    crs: _na.WorkingCrs = inject()
    turbines: AdaptedTurbinesAsset = inject()
    sunpos: SunposAsset = inject()
    receivers: _na.Receivers = inject()
    elevation: _na.Elevation = inject()
    receiver_groups: _na.ReceiverGroups = inject()

    dask_cluster: common_assets.DaskCluster = inject()
    config: ConfigAsset = inject()

    @override
    def make(self):
        turbines = self.turbines.d
        receivers = self.receivers.d
        sun = self.sunpos.d
        daymask = (sun['altitude_rad'] >= np.deg2rad(3.0)).compute()
        # print("DAYMASK")
        # print(daymask)
        # sun_day = sun.where(daymask, drop=True)
        sun_day = sun.isel(time=daymask).chunk(time=1440*3)
        # print("SUN_DAY")
        # print(sun_day)
        # print(sun_day['time'])

        ds = xr.Dataset()

        ds['v_receiver_hub'] = (
            turbines['position'] - receivers['position']
        ).astype(np.float32, copy=False)

        # component of r along sun line (ray parameter lambda)
        # (time,receiver,turbine)
        ds['L_tik'] = ds['v_receiver_hub'].dot(sun_day['v_sun'], 'spatial')

        # perpendicular miss dinstance of sun ray from rotor center
        ds['d_sq'] = (ds['v_receiver_hub']**2).sum('spatial') - ds['L_tik']**2
        ds['d_sq'] = xr.where(ds['d_sq'] > 0, ds['d_sq'], 0)  # guard tiny negatives

        # Determine whether shaded or not
        # ds['sun_ok'] = sun['altitude_rad'] >= np.deg2rad(3.0)
        ds['front_ok'] = ds['L_tik'] > 0
        ds['disk_ok'] = ds['d_sq'] <= turbines['rotor_radius_m']**2
        ds['size_ok'] = ds['L_tik'] <= turbines['L_max']

        ds['shadow_per_turbine'] = ds['front_ok'] & ds['disk_ok'] & ds['size_ok']
        ds['shadow'] = ds['shadow_per_turbine'].any(dim="turbine")

        # duration of each sample in minutes
        _delta: pd.Timedelta = ds.indexes["time"][1] - ds.indexes["time"][0]
        _minutes_per_sample = _delta / pd.Timedelta("1min")

        # Daily totals (number of shadowed samples per day)
        _daily_samples = ds['shadow'].resample(time="1D").sum().rename(time="date")  # dims: (receiver, date)
        ds['daily_minutes'] = _daily_samples * _minutes_per_sample
        ds['max_daily_minutes'] = ds['daily_minutes'].max("date")
        ds['annual_minutes'] = ds['daily_minutes'].sum("date")

        ds['shadow_days'] = ds['shadow'].resample(time="1D").any().rename(time="date")
        ds['annual_shadow_days'] = ds['shadow_days'].sum("date")

        ds['exceeds_daily_30min'] = ds['daily_minutes'] > 30
        ds['exceeds_annual_30h']  = ds['annual_minutes'] > 30*60

        # split the result back into receiver groups and reattach coordinates
        receiver_groups: dict[str, xr.Dataset] = dict()
        start = 0
        for name, group in self.receiver_groups.d.items():
            res = ds.isel(receiver=slice(start, start + group.sizes['receiver']))
            if 'receiver' in group.coords:
                res = res.assign_coords(receiver=group.coords['receiver'])
            receiver_groups[name] = res
            start += group.sizes['receiver']
        assert start == ds.sizes['receiver']

        return ShadowSimulationAsset(
            result=ds,
            receiver_groups=receiver_groups
        )
