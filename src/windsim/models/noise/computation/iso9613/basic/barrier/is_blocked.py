from typing import Any, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from trimesh import Trimesh


def is_blocked_mapblock(turbine_pos: xr.DataArray, receiver_pos: xr.DataArray, barriers: Trimesh) -> xr.DataArray:
    return cast(xr.DataArray, xr.apply_ufunc(
        is_blocked_ufunc, turbine_pos, receiver_pos, barriers,
        input_core_dims=[['turbine', 'spatial'], ['receiver', 'spatial'], []],
        output_core_dims=[['turbine', 'receiver']]
    ))


def is_blocked_ufunc(turbines: npt.NDArray, receivers: npt.NDArray, barriers: Trimesh, ray_spacing: float = 10):
    """Determines for each turbine-receiver pair whether the direct path is blocked by barriers.

    Parameters
    ----------
    turbine_pos : (n,3)
        Turbine positions.
    receiver_pos : (m,3)
        Receiver positions.

    Returns
    --------
    is_blocked : (n,m)
        Boolean array indicating whether the direct path is blocked.
    """
    # Stack all but last dimension. Requires manual broadcasting.
    # Dims: ..., turbine, receiver, offset, spatial
    print("Turbines:", len(turbines), "receivers:", len(receivers))
    return np.empty((len(turbines), len(receivers)), dtype=np.bool_)
    z_offset = np.arange(0, 1000, ray_spacing)
    turbines, receivers, z_offset = np.broadcast_arrays(
        turbines[..., :, np.newaxis, np.newaxis, :],
        receivers[..., np.newaxis, :, np.newaxis, :],
        z_offset[..., np.newaxis, np.newaxis, :, np.newaxis]  # dummy dimension at end
    )
    if turbines.size == 0:
        return np.empty((0,0), dtype=np.bool_)
    z_offset = z_offset[..., 0]  # remove dummy dimension
    
    origins = turbines.copy()
    origins[..., 2] += z_offset

    # One direction vector per pair. Same for each offset.
    # Dims: ..., turbine, receiver, offset, spatial

    # Stack it
    origins_stacked = origins.reshape(-1, 3)
    directions_stacked = (receivers - turbines).reshape(-1, 3)
    distances_stacked = np.linalg.norm(directions_stacked, axis=-1)


    hits, index_ray, _ = barriers.ray.intersects_location(origins_stacked, directions_stacked, multiple_hits=False)
    origins_to_hits = hits - origins_stacked[index_ray]
    hit_distances = np.linalg.norm(origins_to_hits, axis=-1)
    is_valid = hit_distances < distances_stacked[index_ray]

    is_blocked = np.full(len(origins_stacked), np.False_)
    is_blocked[index_ray[is_valid]] = np.True_

    is_blocked = is_blocked.reshape(turbines.shape[:-1])

    # collapse last dimension
    return is_blocked.any(axis=-1)
    exit()
    # is_blocked: npt.NDArray = barriers.ray.intersects_first(origins_stacked, directions_stacked) > -1

    is_blocked = is_blocked.reshape(turbines.shape[:-1])
    
    # collapse last dimension
    return is_blocked.any(axis=-1)

    # # Unstack to restore original dimensions
    # return is_blocked.reshape(turbines.shape[:-1])
