from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
import shapely.ops
import trimesh
import xarray as xr
from shapely import LineString, Polygon
from shapely.errors import GEOSException
from trimesh import Trimesh
from trimesh.path import Path2D, Path3D

# from ..render_3d import render_3d
from ..align_plane import align_plane_transform
from ..plot_2d import plot_2d
from ..utils import PATHS, polygons_to_points, section_to_polygons
from .get_path import get_path
from .shared import Z_AXIS, calc_from_parts

log = logging.getLogger(__name__)


def D_Z_general_ufunc(
    turbine: npt.NDArray,
    receiver: npt.NDArray,
    frequencies: npt.NDArray,
    barriers: Trimesh
) -> npt.NDArray:
    """General computation of D_Z.

    This function is vectorized through `np.vectorize`, which is essentially a Python for loop,
    and is thus very slow."""
    func = np.vectorize(
        _calc_unvectorized,
        signature='(3),(3),(f),()->(f)',
        otypes=[np.float64]
    )

    return func(turbine, receiver, frequencies, barriers)


def _calc_unvectorized(
    turbine: npt.NDArray,
    receiver: npt.NDArray,
    frequencies: npt.NDArray,
    barriers: Trimesh
) -> npt.NDArray:
    """Computes barrier attenuation over the top edge for a single turbine-receiver pair.


    ASSUMPTION: There are no obstacles directly above turbine or receiver.
    We can thus filter the polygons to only include edges that are between turbine and receiver.


    # STEP 1: Only consider polygons between turbine and receiver
    # STEP 2: Check if direct path is blocked.
    #           - if no: count number of edges. Do stuff.
    # STEP 3: Make hull over polygon parts above direct line
    # STEP 4: Remove direct line segment from hull
    
    Parameters
    ---------
    turbine : (3)
        Turbine position.
    receiver : (3)
        Receiver position.
    frequencies : (f)
        Octave frequencies.
    barriers
        Barriers.

    Returns
    -------
    D_Z : (f)
        Barrier attenuation over top edge for the turbine-receiver pair and each octave frequency.
    """
    s, r = turbine, receiver

    # determine vectors
    vec_sr = r - s
    normal = np.cross(vec_sr, Z_AXIS)

    # Slice and transform EV plane, such that receiver is right from turbine.
    flatten = align_plane_transform(normal=normal, point=s, align_y=Z_AXIS, direction_x=vec_sr)
    section_3d = cast(Path3D | None, barriers.section(plane_normal=normal, plane_origin=s))
    try:
        polygons = section_to_polygons(section_3d, flatten) if section_3d is not None else []
    except (ValueError, GEOSException) as e:
        print("Caught error:")
        print(e)
        return np.full(len(frequencies), np.nan)

    s_flat, r_flat = trimesh.transform_points([s, r], flatten)[:, :2]

    return _calc_unvectorized_flat(s_flat, r_flat, polygons, frequencies)


def _calc_unvectorized_flat(s, r, polygons: list[Polygon], frequencies: npt.NDArray) -> npt.NDArray:
    """Compute D_Z in the plane.

    Parameters
    ---------
    s : (2,)
        Tturbine position in EV plane.
    r : (2,)
        Receiver position in EV plane.
    polygons
        Polygons that are fully between `s` and `r`.
    frequencies
        Octave frequencies.

    # Case 1: Intersection is empty.
    # Case 2: Intersection is not empty, but the direct path is unblocked
    # Case 3: Intersection is not empty and the direct path is blocked
    """
    line = LineString([s, r])
    d = np.linalg.norm(r - s)

    # Filter polygons to those fully between turbine and receiver.
    # This is okay because there cannot be barriers above them.
    box = shapely.box(s[0], -1e10, r[0], 1e10)
    tree = shapely.STRtree(polygons)
    polygons_between = list(tree.geometries[tree.query(box, 'contains')])

    # Check if direct path is blocked.
    is_blocked = tree.query(line, 'crosses').size > 0

    # Determine path length difference z.

    if not polygons_between:
        # There are no barriers between turbine and receiver.
        return calc_from_parts(
            z=0,
            e=0,
            K_met=1,
            frequencies=frequencies
        )
    
    if not is_blocked:
        # There are barriers between turbine and receiver, but they are all below the direct line.
        # Compute negative path length difference for all polygon edges.
        edges = polygons_to_points(polygons_between)
        d_ss, d_sr = np.linalg.norm([edges - s, r - edges], axis=-1)
        z_n = -np.abs((d_ss + d_sr) - d)
        return calc_from_parts(
            z=np.max(z_n),
            e=0,
            K_met=1,
            frequencies=frequencies
        )

    # There are barriers between turbine and receiver that block the direct line.
    path = get_path(s, r, polygons_between)

    # compute e as the path length between first and last diffracting edge
    if len(path.coords) > 3:
        e = LineString(path.coords[1:-1]).length
    else:
        e = 0
    d_ss = LineString(path.coords[:2]).length
    d_sr = LineString(path.coords[-2:]).length
    z = (d_ss + d_sr + e) - d

    if z > 0:
        K_met = np.exp(
            -(1/2000) * np.sqrt(
                (d_ss * d_sr * d)
                / (2*z)
            )
        )
    else:
        K_met = 1

    return calc_from_parts(
        z=z,
        e=e,
        K_met=K_met,
        frequencies=frequencies
    )
