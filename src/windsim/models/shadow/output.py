import xarray as xr
from .model import ShadowSimulationAsset, ShadowSimulationRecipe
from planner import Asset, DataAsset, Recipe, inject
from windsim.models.noise import assets as noise_assets

import logging
from typing import Any, cast
import time
from dataclasses import dataclass
from dask.array import compute as da_compute  # type: ignore


log = logging.getLogger(__name__)


@dataclass
class ShadowResultAsset(Asset):
    grid: xr.Dataset | None
    receivers: xr.Dataset | None


class ShadowResultRecipe(Recipe):
    _makes = ShadowResultAsset

    config: noise_assets.Config = inject()
    simulation: ShadowSimulationAsset = inject()
    receiver_groups: noise_assets.ReceiverGroups = inject()

    def make(self):
        # Input variables that should be computed. Must at least contain 'position', which is e.g. used in plotting.
        input_vars = ['position']

        # Result variables that should be computed
        result_vars = [
            "annual_minutes",
            "annual_shadow_days",
            "max_daily_minutes"
        ]

        # Select what to compute
        tasks: dict[str, Any] = {}
        if 'grid' in self.simulation.receiver_groups:
            tasks['grid_input'] = self.receiver_groups.d['grid'][input_vars]
            x = self.simulation.receiver_groups['grid'][result_vars]
            if 'timeslice' in x.dims:
                x = x.squeeze('timeslice')
            tasks['grid_result'] = x
        if 'normal' in self.simulation.receiver_groups:
            tasks['normal_input'] = self.receiver_groups.d['normal'][input_vars]
            x = self.simulation.receiver_groups['normal'][result_vars]
            if 'timeslice' in x.dims:
                x = x.squeeze('timeslice')
            tasks['normal_result'] = x


        # Execute computation
        _compute_start = time.perf_counter()

        computed = dict(zip(
            tasks.keys(),
            cast(tuple[xr.Dataset,...], da_compute(*tasks.values()))
        ))

        _compute_duration = round(time.perf_counter() - _compute_start, 2)
        log.debug(f"Computing results took {_compute_duration} seconds")


        # Process
        log.debug("Processing computed results")

        # Restructure and combine the input and result data for further processing
        if 'grid' in self.simulation.receiver_groups:
            grid_restructured = restructure_grid_data(computed['grid_input'], computed['grid_result'])
        else:
            grid_restructured = None

        if 'normal' in self.simulation.receiver_groups:
            normal_restructured = restructure_normal_data(computed['normal_input'], computed['normal_result'])
            # print(normal_restructured[plot_var])
        else:
            normal_restructured = None

        return ShadowResultAsset(
            grid=grid_restructured,
            receivers=normal_restructured
        )


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
