from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from typing import Any, Literal, TypedDict, cast, overload

import geopandas as gpd
import numpy as np
import pyproj
import shapely
import trimesh
import xarray as xr
from shapely import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry import shape
from trimesh import Trimesh

from ..config import ConfigData


log = logging.getLogger(__name__)


@overload
def _as_fixed_str(obj: xr.DataArray, *, errors: Literal['raise', 'ignore'] = 'raise') -> xr.DataArray: ...
@overload
def _as_fixed_str(obj: xr.Dataset, data_vars: Collection[str], *, errors: Literal['raise', 'ignore'] = 'raise') -> xr.Dataset: ...
def _as_fixed_str(
    obj: xr.DataArray | xr.Dataset,
    data_vars: Collection[str] | None = None,
    *,
    errors: Literal['raise', 'ignore'] = 'raise'
) -> xr.DataArray | xr.Dataset:
    """Pandas stores strings as dtype `object`,
    but storing them as dtype `str_` with a fixed length is more efficient."""
    obj = obj.copy()
    if isinstance(obj, xr.DataArray):
        return obj.astype(np.str_)
    else:
        assert data_vars is not None
        for data_var in data_vars:
            if errors == 'ignore' and data_var not in obj:
                continue
            obj[data_var] = obj[data_var].astype(np.str_)
        return obj


@overload
def _as_dtype(obj: xr.DataArray, dtype: Any, *, errors: Literal['raise', 'ignore'] = 'raise') -> xr.DataArray: ...
@overload
def _as_dtype(obj: xr.Dataset, dtype: Any, data_vars: Collection[str], *, errors: Literal['raise', 'ignore'] = 'raise') -> xr.Dataset: ...
def _as_dtype(
    obj: xr.DataArray | xr.Dataset,
    dtype: Any,
    data_vars: Collection[str] | None = None,
    *,
    errors: Literal['raise', 'ignore'] = 'raise'
) -> xr.DataArray | xr.Dataset:
    obj = obj.copy()
    if isinstance(obj, xr.DataArray):
        return obj.astype(dtype)
    else:
        assert data_vars is not None
        for data_var in data_vars:
            if errors == 'ignore' and data_var not in obj:
                continue
            obj[data_var] = obj[data_var].astype(dtype)
        return obj


# def environment(asset: assets.GisObjectsJson, working_crs: pyproj.CRS) -> xr.Dataset:
#     """Create environment description from JSON"""
#     df = gpd.GeoDataFrame(asset.data)
#     # set lat/long
#     df['geometry_wgs'] = gpd.GeoSeries(df["geometry"].apply(shape), crs=CRS.WGS84)  # type: ignore
#     df.drop(columns=["latitude", "longitude"], inplace=True)
#     # set working CRS
#     df['geometry'] = cast(gpd.GeoSeries, df['geometry_wgs']).to_crs(working_crs)
#     df.set_geometry('geometry', inplace=True)
    
#     return xr.Dataset.from_dataframe(df)


def layout(environment: xr.Dataset, *, config: ConfigData) -> tuple[MultiPolygon, list[float]]:
    buildings = environment.sel(index=environment['object_class'] == 'residential_building')

    # DEBUG: filter barriers
    barrier_filter = config.debug.filter_barriers
    if barrier_filter is not None:
        if barrier_filter.postcode is not None:
            buildings = buildings.sel(index=buildings['postcode'] == barrier_filter.postcode)

    polygons_raw = buildings['geometry'].to_numpy()
    heights_raw = buildings['building_height_m'].to_numpy()

    # DEBUG: adjust heights by random offset
    _jitter = config.debug.jitter_building_heights
    if _jitter is not None:
        mean, max_dev = _jitter.average_offset, _jitter.max_offset_deviation
        z_score = 2  # 95.4 % interval
        std_dev = max_dev / z_score
        rand_offset = (
            config.rng.normal(mean, std_dev, size=len(heights_raw))
            .clip(mean - max_dev, mean + max_dev)
        )
        heights_raw += rand_offset

    # validate polygons
    polygons: list[Polygon] = []
    heights: list[float] = []
    for p, h in zip(polygons_raw, heights_raw):
        # ensure the polygons are valid
        p = shapely.make_valid(p)
        if isinstance(p, Polygon):
            polygons.append(p)
            heights.append(h)
        elif isinstance(p, MultiPolygon):
            ps = p.geoms
            polygons += ps
            heights += [h] * len(ps)
        elif isinstance(p, GeometryCollection):
            ps = [x for x in p.geoms if isinstance(x, Polygon)]
            polygons += ps
            heights += [h] * len(ps)
        else:
            raise ValueError(f"Polygon has invalid type after validation: {type(p)}")
    assert len(polygons) == len(heights)
    assert all(isinstance(p, Polygon) for p in polygons)

    return MultiPolygon(polygons), heights


def barriers_3d(polygons: MultiPolygon, heights: list[float]) -> trimesh.Trimesh:
    # convert 2D polygons to 3D meshes
    meshes: list[trimesh.Trimesh] = []
    for p, h in zip(polygons.geoms, heights):
        meshes.append(
            # avoid manifold engine until https://github.com/mikedh/trimesh/issues/2266 is merged
            trimesh.creation.extrude_polygon(p, h, engine="triangle")
        )
    
    # Move everything towards the origin.
    # This seems to be necessary for boolean union to work correctly.
    concatenation = cast(Trimesh, trimesh.util.concatenate(meshes))
    translation = -concatenation.center_mass
    for mesh in meshes:
        mesh.apply_translation(translation)
    
    # combine meshes and move back to original coords
    mesh = cast(Trimesh, trimesh.boolean.union(cast(Any, meshes)))
    mesh.apply_translation(-translation)

    return mesh
