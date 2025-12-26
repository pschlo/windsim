from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
import xarray as xr
from trimesh import Trimesh

from .general import D_Z_general_ufunc
from .unblocked import D_Z_unblocked_ufunc

log = logging.getLogger(__name__)


def D_Z_top_mapblock(
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
        output_core_dims=[['turbine', 'receiver', 'frequency']]
    ))


def D_Z_ufunc(
    turbines: npt.NDArray,
    receivers: npt.NDArray,
    frequencies: npt.NDArray,
    barriers: Trimesh,
) -> npt.NDArray:
    """Computes barrier attenuation over the top for all turbine-receiver pairs.
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

    # result_stacked = np.empty((len(pairs_stacked), len(frequencies)))

    # result_stacked[~blocked_mask_stacked] = D_Z_unblocked(frequencies)

    # if pairs_blocked_stacked.size > 0:
    #     result_stacked[blocked_mask_stacked] = D_Z_blocked_ufunc(
    #         pairs_blocked_stacked[:,0], pairs_blocked_stacked[:,1], frequencies, barriers
    #     )

    # Unstack
    return result_stacked.reshape((*turbines.shape[:-1], len(frequencies)))


