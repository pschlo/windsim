from __future__ import annotations

import logging
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
import shapely.ops
import trimesh
import xarray as xr
from shapely import LineString, MultiPolygon, Polygon, STRtree
from trimesh import Trimesh
from trimesh.path import Path2D, Path3D

# from ..render_3d import render_3d
from ..align_plane import align_plane_transform
from ..plot_2d import plot_2d
from ..utils import PATHS, section_to_polygons, slice_barriers
from .get_paths import get_paths
from .shared import calc_from_parts

ORIGIN = np.array([0,0,0])
X_AXIS = np.array([1,0,0])
Y_AXIS = np.array([0,1,0])
Z_AXIS = np.array([0,0,1])


def D_Z_general_ufunc(
    turbine: npt.NDArray,
    receiver: npt.NDArray,
    frequencies: npt.NDArray,
    barriers: Trimesh
) -> npt.NDArray:
    """General computation of D_Z.

    This function is vectorized through `np.vectorize`, which is essentially a Python for loop,
    and is thus very slow."""
    if turbine.size == 0:
        return np.zeros((len(turbine)*len(receiver),2,len(frequencies)), dtype=np.float64)

    func = np.vectorize(
        _calc_unvectorized,
        signature='(3),(3),(f),()->(2,f)',
        otypes=[np.float64]
    )

    return func(turbine, receiver, frequencies, barriers)


def _calc_unvectorized(
    turbine: npt.NDArray,
    receiver: npt.NDArray,
    frequencies: npt.NDArray,
    barriers: Trimesh
) -> npt.NDArray:
    """Computes barrier attenuation along sides for a turbine-receiver pair whose direct path
    is blocked by barriers.

    Parameters
    --------
    turbine : (3)
        Turbine position.
    receicer : (3)
        Receiver position.
    frequency : (f)
        Octave frequencies.
    barriers
        Barriers.
    
    Returns
    --------
    D_Z : (2,f)
        Barrier attenuation for the turbine-receiver pair and each octave frequency.
        First axis is (left, right).
    """
    s, r = turbine, receiver

    # determine vectors
    vec_sr = r-s
    _normal_EV = np.cross(vec_sr, Z_AXIS)
    normal = np.cross(vec_sr, _normal_EV)

    # Slice and transform EL plane, such that receiver is right from turbine.
    flatten = align_plane_transform(normal=normal, point=s, align_y=Y_AXIS, direction_x=vec_sr)
    section_3d = cast(Path3D | None, barriers.section(plane_normal=normal, plane_origin=s))
    polygons = section_to_polygons(section_3d, flatten) if section_3d is not None else []
    s_flat, r_flat = trimesh.transform_points([s, r], flatten)[:, :2]

    return _calc_unvectorized_flat(s_flat, r_flat, polygons, frequencies)


def _calc_unvectorized_flat(s, r, polygons: list[Polygon], frequencies: npt.NDArray) -> npt.NDArray:
    """Compute D_Z in the plane.
    
    Parameters
    ---------
    s : (2,)
        Tturbine position in EL plane.
    r : (2,)
        Receiver position in EL plane.
    polygons
        Polygons in the EL plane.
    frequencies
        Octave frequencies.

    Returns
    --------
    D_Z : (2,f)
    """

    line = LineString([s, r])
    d = np.linalg.norm(r - s)
    tree = STRtree(polygons)

    # Check if direct path is blocked.
    is_blocked = tree.query(line, 'crosses').size > 0

    if not is_blocked:
        # Direct line is unblocked.
        v = calc_from_parts(
            z=0,
            e=0,
            K_met=1,
            frequencies=frequencies
        )
        return np.vstack([v, v])
    
    # There are barriers between turbine and receiver that block the direct line.
    # Determine paths.
    left_path, right_path = get_paths(s, r, polygons, tree)

    def process_path(path: LineString | None) -> npt.NDArray:
        if path is None:
            return np.full(len(frequencies), np.nan)

        if len(path.coords) > 3:
            e = LineString(path.coords[1:-1]).length
        else:
            e = 0
        d_ss = LineString(path.coords[:2]).length
        d_sr = LineString(path.coords[-2:]).length
        z = (d_ss + d_sr + e) - d

        K_met = 1

        return calc_from_parts(
            z=z,
            e=e,
            K_met=K_met,
            frequencies=frequencies
        )
    
    return np.vstack([process_path(left_path), process_path(right_path)])
