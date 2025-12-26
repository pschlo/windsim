from collections.abc import Iterable
from typing import Any, Literal, cast

import numpy as np
from scipy.spatial import ConvexHull
from shapely import LineString, Polygon, STRtree

from ..utils import hull_path, polygons_to_points, validate_path


def get_path(s, r, polygons: Iterable[Polygon]) -> LineString:
    """
    Constructs the vertical path for a turbine-receiver pair where the direct line is blocked.

    Parameters
    ---------
    s
        Turbine.
    r
        Receiver.
    polygons
        Collection of polygons that lie fully between turbine and receiver.

    ASSUMPTION: r is right of s.
    """
    line = LineString([s, r])
    vec_sr = r-s

    # Construct path that goes above all polygons.
    # Only add parts above the line segment.
    points = polygons_to_points(polygons)
    points_above = points[np.cross(vec_sr, points) > 0]
    assert len(points_above) > 0

    hull = ConvexHull(np.vstack([s, r, points_above]))
    path = hull_path(hull, line)
    path = validate_path(path, s, r)
    
    return path
