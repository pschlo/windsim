import xarray as xr
from dataclasses import dataclass
import numpy as np
from dask.array import compute as da_compute  # type: ignore
import logging
from typing import override, cast, Any
from datetime import datetime
from contextlib import contextmanager
import time

from planner import Asset, Recipe, DataAsset, inject, assets
from ..input import assets as noise_assets
from ..model import NoiseSimulationAsset
from . import plotting
from ..config import ConfigAsset


log = logging.getLogger(__name__)


@dataclass
class NoiseOutputAsset(Asset):
    receiver_sound_pressure_levels: xr.DataArray | None
    grid_sound_pressure_levels: xr.DataArray | None


class NoiseOutputRecipe(Recipe[NoiseOutputAsset]):
    _makes = NoiseOutputAsset
    _dir = 'noise_output'

    config: ConfigAsset = inject()
    result: NoiseSimulationAsset = inject()
    area: noise_assets.Area = inject()
    crs: noise_assets.WorkingCrs = inject()
    receiver_groups: noise_assets.ReceiverGroups = inject()
    turbines: noise_assets.FullTurbines = inject()
    elevation: noise_assets.Elevation = inject()

    @override
    @contextmanager
    def make(self):
        """Processes the simulation result. Lazy data will be computed as required."""
        
        log.debug("Computing results")
        conf = self.config.d

        # Just choose what should be calculated and plotted.
        # TODO: make it configurable what should be computed and contained in outputs
        main_result_var = 'L_r' if conf.computation.ta_laerm else 'L_AT_LT'

        # Input variables that should be computed. Must at least contain 'position', which is e.g. used in plotting.
        input_vars = ['position']
        # Result variables that should be computed
        result_vars = [main_result_var]
        # Variable that should be plotted
        plot_var = main_result_var


        # Select what to compute
        # TODO: How to handle multiple timeslices?
        tasks: dict[str, Any] = {}
        if 'grid' in self.result.receiver_groups:
            tasks['grid_input'] = self.receiver_groups.d['grid'][input_vars]
            x = self.result.receiver_groups['grid'][result_vars]
            if 'timeslice' in x.dims:
                x = x.squeeze('timeslice')
            tasks['grid_result'] = x
        if 'normal' in self.result.receiver_groups:
            tasks['normal_input'] = self.receiver_groups.d['normal'][input_vars]
            x = self.result.receiver_groups['normal'][result_vars]
            if 'timeslice' in x.dims:
                x = x.squeeze('timeslice')
            tasks['normal_result'] = x


        # Execute computation
        _compute_start = time.perf_counter()

        computed = dict(zip(
            tasks.keys(),
            cast(tuple[xr.Dataset,...], da_compute(*tasks.values()))
        ))

        _compute_duration = round(time.perf_counter() - _compute_start, conf.time_decimal_places)
        log.debug(f"Computing results took {_compute_duration} seconds")


        # Process
        log.debug("Processing computed results")

        # Restructure and combine the input and result data for further processing
        if 'grid' in self.result.receiver_groups:
            grid_restructured = restructure_grid_data(computed['grid_input'], computed['grid_result'])
        else:
            grid_restructured = None

        if 'normal' in self.result.receiver_groups:
            normal_restructured = restructure_normal_data(computed['normal_input'], computed['normal_result'])
            # print(normal_restructured[plot_var])
        else:
            normal_restructured = None


        # Store restructured data to file
        if conf.output.file is not None:
            log.debug("  Saving results to file")
            timestamp = datetime.now().astimezone().strftime(r'%Y-%m-%d_%H-%M-%S_UTC%z')
            if normal_restructured is not None:
                normal_restructured.to_netcdf(conf.output.file.folder / f'specific-receivers_export_{timestamp}.nc')
            if grid_restructured is not None:
                grid_restructured.to_netcdf(conf.output.file.folder / f'grid-receivers_export_{timestamp}.nc')

        # Plot restructured data
        if conf.output.map is not None:
            log.debug("  Plotting results")
            plotting.plot(
                plot_variable=plot_var,
                grid_restructured=grid_restructured,
                normal_restructured=normal_restructured,
                area=self.area.d,
                working_crs=self.crs.d,
                # features=result.input.layout.d[0],
                elevation=self.elevation.d,
                turbines=self.turbines.d,
                config=conf,
                folder=self.workdir
            )

        try:
            yield NoiseOutputAsset(
                receiver_sound_pressure_levels=normal_restructured[main_result_var] if normal_restructured is not None else None,
                grid_sound_pressure_levels=grid_restructured[main_result_var] if grid_restructured is not None else None
            )
        finally:
            print("CLEANING UP OUTPUT")


def restructure_grid_data(input: xr.Dataset, result: xr.Dataset) -> xr.Dataset:
    # Reshape such that the positions are turned into dimensions X and Y with corresponding coordinates.
    result_xy = (
        result
        .assign_coords(
            x=input['position'].sel(spatial='x'),
            y=input['position'].sel(spatial='y')
        )
        .set_xindex(['x', 'y'])
        .unstack('receiver')
    )
    return result_xy


def restructure_normal_data(input: xr.Dataset, result: xr.Dataset) -> xr.Dataset:
    return xr.merge([input, result])
