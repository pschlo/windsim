from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import geopandas as gpd
import numpy as np
import pyproj
import shapely
import shapely.ops
from matplotlib.contour import ContourSet
from shapely import (LinearRing, LineString, MultiLineString, MultiPolygon,
                     Point, Polygon)

RingSet: TypeAlias = dict[float, set[LinearRing]]


def compare_contours(
    matplotlib_contours: ContourSet,
    windpro_contours_path: Path | str,
    crs: pyproj.CRS,
    num_linepoints: int
):
    wp_contours = _parse_windpro_contours(
        gpd.read_file(windpro_contours_path).to_crs(crs),
        num_linepoints=num_linepoints
    )
    own_contours = _parse_matplotlib_contours(
        matplotlib_contours,
        num_linepoints=num_linepoints
    )

    print("From own to WindPRO")
    total_res, level_res = _compare_ringsets(own_contours, wp_contours)
    for level, res in level_res.items():
        print(f"  Level {level}: min={res.min_dist}, max={res.max_dist}, avg={res.avg_dist}")
    print(f"Total: min={total_res.min_dist}, max={total_res.max_dist}, avg={total_res.avg_dist}")
    # print()
    # print("From WindPRO to own")
    # total_res, level_res = _compare_ringsets(wp_contours, own_contours)
    # for level, res in level_res.items():
    #     print(f"  Level {level}: min={res.min_dist}, max={res.max_dist}, avg={res.avg_dist}")
    # print(f"Total: min={total_res.min_dist}, max={total_res.max_dist}, avg={total_res.avg_dist}")
    

def _parse_windpro_contours(gdf: gpd.GeoDataFrame, num_linepoints: int) -> RingSet:
    contours_windpro: RingSet = {}
    for level, group in gdf.groupby('NOISE')['geometry']:
        assert isinstance(level, (int, float))

        lines = shapely.ops.linemerge(list(group))
        if isinstance(lines, LineString):
            lines = [lines]
        elif isinstance(lines, MultiLineString):
            lines = list(lines.geoms)
        else:
            assert False

        lines = {interp_line(line, num_linepoints) for line in lines}
        rings = {LinearRing(line) for line in lines if line.is_closed}
        if len(rings) < len(lines):
            print(f"Skipping level {level} because some lines are not rings")
        else:
            contours_windpro[level] = rings

    return contours_windpro


def _parse_matplotlib_contours(cs: ContourSet, num_linepoints: int) -> RingSet:
    contours_own: RingSet = {}
    for level, segs in zip(cs.levels, cs.allsegs):
        lines = {LineString(seg) for seg in segs if len(seg) > 0}
        lines = {interp_line(line, num_linepoints) for line in lines}
        rings = {LinearRing(line) for line in lines if line.is_closed}
        if len(rings) < len(lines):
            print(f"Skipping level {level} because some lines are not rings")
        else:
            contours_own[level] = rings

    return contours_own


def interp_line(line: LineString, n: int):
    """Interpolate line to `n` evenly spaced points."""
    distances = np.linspace(0, line.length, n)
    points = [line.interpolate(distance) for distance in distances]
    # prevent numerical inaccuracies
    points[0] = line.coords[0]
    points[-1] = line.coords[-1]
    return LineString(points)


@dataclass
class ComparisonResult:
    min_dist: float
    max_dist: float
    avg_dist: float


def _compare_ringsets(ringset: RingSet, other_ringset: RingSet) -> tuple[ComparisonResult, dict[float, ComparisonResult]]:
    results: dict[float, ComparisonResult] = {}
    for level, rings in ringset.items():
        if level not in other_ringset:
            print(f"Skipping level {level}: Level does not exist in other contour set")
            continue
        other_rings = other_ringset[level]
        if len(rings) != len(other_rings):
            print(f"Skipping level {level}: Not the same number of contour rings")
        vertices = np.array([Point(p) for ring in rings for p in ring.coords])

        other_lines = MultiLineString(list(other_rings))
        other_polygons = MultiPolygon([Polygon(ring) for ring in other_rings])
        distances = other_lines.distance(vertices)
        distances[other_polygons.contains(vertices)] *= -1

        results[level] = ComparisonResult(
            min_dist=min(distances),
            max_dist=max(distances),
            avg_dist=sum(distances) / len(distances)
        )

    total_result = ComparisonResult(
        min_dist=min(r.min_dist for r in results.values()),
        max_dist=max(r.max_dist for r in results.values()),
        avg_dist=sum(r.avg_dist for r in results.values()) / len(results)
    )
    return total_result, results
