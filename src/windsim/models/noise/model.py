from __future__ import annotations
from pathlib import Path
from typing import override
import logging
import xarray as xr
import time
from dataclasses import dataclass

from planner import Asset, Recipe, inject, DataAsset

from windsim.common import assets as common_assets
from windsim.models.noise.input import assets as _na
from windsim.models.noise.computation import iso9613 as iso
from .config import ConfigAsset


log = logging.getLogger(__name__)


@dataclass
class NoiseSimulationAsset(Asset):
    result: xr.Dataset
    receiver_groups: dict[str, xr.Dataset]


class NoiseSimulationRecipe(Recipe[NoiseSimulationAsset]):
    _makes = NoiseSimulationAsset

    turbine_types: _na.TurbineModels = inject()
    area: _na.Area = inject()
    crs: _na.WorkingCrs = inject()
    turbines: _na.FullTurbines = inject()
    receivers: _na.Receivers = inject()
    timeslice_durations: _na.TimesliceDurations = inject()
    atmospheric_coefficient: _na.AtmosphericCoefficient = inject()
    exposure_times: _na.ExposureTimes = inject()
    coarseness: _na.Coarseness = inject()
    sound_power_levels: _na.SoundPowerLevels = inject()
    elevation: _na.Elevation = inject()
    frequencies: _na.Frequencies = inject()
    a_weighting: _na.AWeighting = inject()
    receiver_groups: _na.ReceiverGroups = inject()

    dask_cluster: common_assets.DaskCluster = inject()
    config: ConfigAsset = inject()

    @override
    def make(self):
        """
        Creates the Dask task graph for the simulation results.
        
        Parameters
        -----------
        _input
            The prepared input data.

        Returns
        --------
        result
            The lazy simulation result. Under the hood, this contains Dask task graphs
            that contain information on how to actually compute the results.
        """
        log.debug("Running noise simulation")
        conf = self.config.d

        interim = conf.computation.interim
        alternative = conf.computation.alternative_method
        ta_laerm = conf.computation.ta_laerm

        if interim and alternative:
            raise RuntimeError("Interimsverfahren does not allow the ISO 9613-2 alternative method")

        log.info(f"Building computation graph")
        log.info("")
        log.info(f'Simulation input:')
        log.info(f'  - {self.turbines.d.sizes["turbine"]} turbines')
        log.info(f'  - {self.receivers.d.sizes["receiver"]} receivers')
        log.info(f'  - {self.timeslice_durations.d.sizes["timeslice"]} time slices')

        # self.turbines.d['position'] = self.turbines.d['position'].expand_dims(time=[1,2,3,4])
        # self.turbines.d['position'] = self.turbines.d['position'].expand_dims(time=4)

        _graph_start = time.perf_counter()

        ds = xr.Dataset()

        # perform simulation steps one by one

        ds = iso.basic.distance(ds, self.turbines.d, self.receivers.d)
        ds = iso.basic.distance_proj(ds)
        ds = iso.basic.avg_height(ds, self.turbines.d, self.receivers.d)
        ds = iso.basic.A_misc(ds)
        ds = iso.basic.A_div(ds)
        ds = iso.basic.A_atm(ds, self.atmospheric_coefficient.d)

        if interim:
            ds = iso.interim.A_gr_interim(ds)
        elif alternative:
            ds = iso.basic.A_gr_alternative(ds, self.turbines.d, self.receivers.d)
        else:
            ds = iso.basic.A_gr(ds, self.turbines.d, self.receivers.d, self.coarseness.d)

        if conf.computation.consider_barriers:
            # ds = iso.basic.is_blocked(
            #     ds,
            #     turbines=self.turbines,
            #     receivers=self.receivers,
            #     barriers=self.barriers_3d,
            # )

            ds = iso.basic.D_Z_top(
                ds,
                self.turbines.d,
                self.receivers.d,
                self.barriers_3d.d,
                xr.DataArray(self.frequencies.d, coords=dict(frequency=self.frequencies.d))
            )

            # ds = iso.basic.D_Z_sides(
            #     ds,
            #     self.turbines,
            #     self.receivers,
            #     self.barriers_3d,
            #     xr.DataArray(self.frequencies, coords=dict(frequency=self.frequencies))
            # )

            ds = iso.basic.A_bar_top(ds)
            # ds = iso.basic.A_bar_sides(ds)
            # ds = iso.basic.A_bar(ds)
            ds['A_bar'] = ds['A_bar_top']
        else:
            ds['A_bar'] = 0

        ds = iso.basic.A_total(ds)
        ds = iso.basic.D_I(ds)
        ds = iso.interim.D_C_interim(ds) if interim else iso.basic.D_C(ds)
        ds = iso.basic.L_fT_DW(ds, self.sound_power_levels.d)

        # either apply TA Lärm or proceed with ISO 9613-2
        if ta_laerm:
            ds = iso.ta_laerm.L_Aeq_k_j(ds, self.sound_power_levels.d, self.a_weighting.d)
            ds = iso.ta_laerm.L_Aeq_j(ds, self.exposure_times.d, self.timeslice_durations.d)
            ds = iso.ta_laerm.L_r(ds, self.timeslice_durations.d)
        else:
            ds = iso.basic.L_AT_DW(ds, self.sound_power_levels.d, self.a_weighting.d)
            ds = iso.interim.C_met_interim(ds) if interim else iso.basic.C_met(ds)
            ds = iso.basic.L_AT_LT(ds)
        
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

        _duration = round(time.perf_counter() - _graph_start, conf.time_decimal_places)
        log.info("")
        log.info(f"Building computation graph took {_duration} seconds")
        # print(ds)


        # ds['L_r'].data.dask.visualize(filename="task-graph.svg")
        # ds['L_r'].data.dask.visualize(filename="task-graph_optimized.svg", optimize_graph=True)

        # print()
        # print()
        # ds.load()
        # print(ds)
        # exit()

        # print(ds['L_Aeq_j'].sel(humidity=70, temperature=10))
        # print(ds['L_r'].sel(humidity=70, temperature=10))
        # print()
        # print(ds['D_omega'])
        # exit()

        # compute L_r
        # print("computing L_r")
        # ds['L_r'].compute()
        # exit()

        # log.debug("Computing")
        # self.receivers['position'].load()
        # ds['L_r'].load()
        # self.receivers['type'].load()
        # print("DONE")

        return NoiseSimulationAsset(
            result=ds,
            receiver_groups=receiver_groups,
        )
