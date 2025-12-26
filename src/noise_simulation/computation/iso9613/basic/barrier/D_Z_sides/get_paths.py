from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
from shapely import (LinearRing, LineString, MultiLineString, MultiPolygon,
                     Point, Polygon)
from shapely.plotting import plot_line, plot_polygon
from shapely.strtree import STRtree

from ..utils import (PropagationPathError, hull_path, hull_ring,
                     polygons_to_points, validate_path)

log = logging.getLogger(__name__)


def get_paths(s, r, polygons: Iterable[Polygon], tree: STRtree) \
    -> tuple[LineString | None, LineString | None]:
    """Constructs the two lateral paths for a turbine-receiver pair where the direct line is blocked.
    
    Parameters
    ----------
    s
        Turbine.
    r
        Receiver.
    polygons
        Collection of polygons.
    tree
        Index tree, used for fast polygon lookup.
    """

    polygons = np.asarray(polygons)
    line = LineString([s, r])
    vec_sr = r - s

    # Construct initial convex hulls from polygons that cross the direct line.
    # Only add points above/below the segment.
    _crossing_polys = tree.geometries[tree.query(line, 'crosses')]
    points = polygons_to_points(_crossing_polys)
    _cross = np.cross(vec_sr, points)
    points_left = points[_cross > 0]
    points_right = points[_cross < 0]
    assert len(points_left) > 0 and len(points_right) > 0

    left_hull = ConvexHull(np.vstack([s, r, points_left]), incremental=True)
    right_hull = ConvexHull(np.vstack([s, r, points_right]), incremental=True)

    # construct the top and bottom propagation paths
    left_path = _get_path(s, r, polygons, tree, left_hull)
    right_path = _get_path(s, r, polygons, tree, right_hull)

    return left_path, right_path


def _get_path(
    s,
    r,
    polygons: npt.NDArray,
    tree: STRtree,
    hull: ConvexHull,
) -> LineString | None:
    """Given a convex hull that represents a single propagation path,
    expand until the path does not cross any barriers.
    
    Returns
    -------
    path
        Propagation path from S to R. `None` if the path does not exist.
    """
    segment_between = LineString([s, r])

    # Check for crossing polygons, but exclude those on the direct line segment
    path = hull_path(hull, segment_between)
    cross_path_indices = tree.query(path, 'crosses')

    while len(cross_path_indices) > 0:
        # add intersecting polygons to hull
        hull.add_points(polygons_to_points(polygons[cross_path_indices]))
        # check for crossing polygons, but exclude those on the direct line segment
        path = hull_path(hull, segment_between)
        cross_path_indices = tree.query(path, 'crosses')

    try:
        path = validate_path(path, s, r)
    except PropagationPathError:
        log.warning("Propagation path does not connect S and R as expected, ignoring the path")
        path = None

    return path
