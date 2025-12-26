from __future__ import annotations

import logging
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
import shapely.ops
import trimesh
import xarray as xr
from trimesh import Trimesh
from trimesh.path import Path2D, Path3D

# from ..render_3d import render_3d
from ..plot_2d import plot_2d
from .general import D_Z_general_ufunc


def D_Z_sides_mapblock(
    turbines: xr.DataArray,
    receivers: xr.DataArray,
    frequencies: xr.DataArray,
    barriers: Trimesh,
):
    return cast(xr.DataArray, xr.apply_ufunc(
        D_Z_ufunc, turbines, receivers, frequencies, barriers,
        input_core_dims=[
            ['turbine', 'spatial'],
            ['receiver', 'spatial'],
            ['frequency'],
            [],
        ],
        output_core_dims=[['turbine', 'receiver', 'side', 'frequency']]
    )).assign_coords(side=['left', 'right'])


def D_Z_ufunc(
    turbines: npt.NDArray,
    receivers: npt.NDArray,
    frequencies: npt.NDArray,
    barriers: Trimesh,
) -> npt.NDArray:
    """Computes barrier attenuation along sides for all turbine-receiver pairs.
    Uses `blocked_mask` to filter out pairs whose direct path is not blocked.
    """
    # Broadcast manually
    turbines, receivers = np.broadcast_arrays(
        turbines[..., :, np.newaxis, :],
        receivers[..., np.newaxis, :, :],
    )
    pairs = np.stack([turbines, receivers], -2)

    # Stack
    pairs_stacked = pairs.reshape(-1, 2, 3)
    # blocked_mask_stacked = blocked_mask.reshape(-1)
    # pairs_blocked_stacked = pairs_stacked[blocked_mask_stacked]

    # Compute result
    result_stacked = D_Z_general_ufunc(
        pairs_stacked[:,0], pairs_stacked[:,1], frequencies, barriers
    )

    # result_stacked = np.empty((len(pairs_stacked), 2, len(frequencies)))
    # result_stacked[~blocked_mask_stacked] = 0
    # if pairs_blocked_stacked.size > 0:
    #     result_stacked[blocked_mask_stacked] = D_Z_general_ufunc(
    #         pairs_blocked_stacked[:,0], pairs_blocked_stacked[:,1], frequencies, barriers
    #     )

    # Unstack
    return result_stacked.reshape((*turbines.shape[:-1], 2, len(frequencies)))
