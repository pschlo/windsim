from __future__ import annotations

import logging
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
import trimesh
import xarray as xr
from trimesh import Trimesh

from . import D_Z_top, is_blocked
from .D_Z_sides import general
# from .render_3d import render_3d
from .plot_2d import plot_2d

log = logging.getLogger(__name__)


"""
    points = trimesh.PointCloud([s, r])
    points_EL_xy_2d = trimesh.points.PointCloud([s_EL_xy_2d, r_EL_xy_2d])
    points_EV_xy_2d = trimesh.points.PointCloud([s_EV_xy_2d, r_EV_xy_2d])



    # unflatten_EL = trimesh.transformations.inverse_matrix(EL_to_xy)

    # ring_3d = np.column_stack([ring.coords, np.zeros(len(ring.coords))])
    # ring_unflattened = trimesh.transform_points(ring_3d, unflatten_EL)

    # ring_path3d = trimesh.load_path(ring_unflattened)
    # assert isinstance(ring_path3d, Path3D)

    # render(barriers, ring_path3d, renderer="trimesh", lines_to_mesh=True)
"""
