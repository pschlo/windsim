import logging
from typing import Any, Literal, Type, cast, overload

import numpy as np
import numpy.typing as npt
import shapely.ops
import trimesh
from scipy.spatial import ConvexHull
from shapely import (LinearRing, LineString, MultiLineString, MultiPolygon,
                     Polygon)
from shapely.errors import GEOSException
from trimesh import Trimesh
from trimesh.path import Path2D, Path3D

log = logging.getLogger(__name__)


def hull_ring(hull: ConvexHull) -> LinearRing:
    return LinearRing(hull.points[hull.vertices])

def hull_path(hull: ConvexHull, segment: LineString) -> LineString:
    """Removes a part from the hull."""
    path = hull_ring(hull).difference(segment)
    assert isinstance(path, LineString | MultiLineString)
    if isinstance(path, MultiLineString):
        path = shapely.ops.linemerge(path)
        assert isinstance(path, LineString)
    return path

def polygons_to_points(polygons) -> npt.NDArray:
    return np.vstack([p.exterior.coords for p in polygons])

# def split(polygons, line: LineString) -> list[Polygon]:
#     _splits = [shapely.ops.split(p, line) for p in polygons]
#     _polys = [poly for split in _splits for poly in split.geoms]
#     assert all(isinstance(p, Polygon) for p in _polys)
#     return cast(list[Polygon], _polys)


class PropagationPathError(Exception):
    pass

def validate_path(path: LineString, s, r) -> LineString:
    """Ensure s is the start and r is the end of the path.

    If the path starts at r and ends at s, reverse it.
    If the start and end are close enough to s and r, explicitly set them to s and r.
    """
    s, r = np.asarray(s), np.asarray(r)
    coords = np.asarray(path.coords)

    if np.allclose(coords[0], s) and np.allclose(coords[-1], r):
        # Path is correctly oriented
        pass
    elif np.allclose(coords[0], r) and np.allclose(coords[-1], s):
        # Path is reversed; reverse it to match s and r
        coords = coords[::-1]
    else:
        raise PropagationPathError("Path does not start or end at the expected points.")

    # Explicitly set the start and end points to s and r
    coords[0], coords[-1] = s, r

    return LineString(coords)


def section_to_polygons(section_3d: Path3D, flatten: npt.NDArray):
    vertices_flat = trimesh.transform_points(section_3d.vertices, flatten)
    section_2d = Path2D(
        entities=section_3d.entities,
        vertices=vertices_flat[:, :2],
        metadata=section_3d.metadata,
        process=False
    )
    polygons_raw = cast(list[shapely.Polygon], section_2d.polygons_full)
    if not polygons_raw:
        return []
    
    # combine overlapping polygons and snap to grid
    combined_polygons = shapely.unary_union(polygons_raw, grid_size=1e-8)
    if isinstance(combined_polygons, Polygon):
        polygons = [combined_polygons]
    elif isinstance(combined_polygons, MultiPolygon):
        polygons = list(combined_polygons.geoms)
    else:
        assert False

    return polygons



def slice_barriers(normal, origin, barriers: Trimesh, flatten: npt.NDArray) -> list[Polygon]:
    """Computes a barrier cross section, transforms it to lie within the xy plane
    and determines the resulting polygons.

    Parameters
    ---------
    origin
        Origin of the section plane.
    normal
        Normal vector of the section plane.
    barriers
        Barriers to cut through.
    flatten
        Transformation matrix that aligns the plane defined by `normal` and `origin` with the xy plane.
    
    Returns
    --------
    polygons
        List of `Polygon` that make up the flattened cross section.
    """

    section_3d = cast(Path3D | None, barriers.section(plane_normal=normal, plane_origin=origin))
    if section_3d is None:
        return []
    vertices_flat = trimesh.transform_points(section_3d.vertices, flatten)
    section_2d = Path2D(
        entities=section_3d.entities,
        vertices=vertices_flat[:, :2],
        metadata=section_3d.metadata,
        process=False
    )
    polygons = cast(list[shapely.Polygon], section_2d.polygons_full)
    
    return polygons




PATHS = dict(top=[], left=[], right=[], flatten_top=[], flatten_lat=[])
