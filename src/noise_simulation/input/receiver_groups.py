import pyproj.aoi
import logging
from typing import override, cast, Any
import pandas as pd
import xarray as xr
import numpy as np
import dask.array.core as da_core
import dask.array.creation as da_creation
import dask.array.wrap as da_wrap

from planner import Recipe, DataAsset, inject
from noise_simulation.coordinate_reference_systems import CRS

from .elevation import ElevationAsset
from .working_crs import WorkingCrsAsset
from .area_of_interest import AreaOfInterestAsset
from .area import AreaAsset, Area
from .chunksize import ChunksizeAsset, Chunksize
from ._utils import _as_fixed_str
from ..config import ConfigAsset
from ..config.sections.input import GridReceivers, NormalReceivers


log = logging.getLogger(__name__)


class ReceiverGroupsAsset(DataAsset[dict]):
    pass


class ReceiverGroupsRecipe(Recipe[ReceiverGroupsAsset]):
    _makes = ReceiverGroupsAsset

    config: ConfigAsset = inject()
    receivers: assets.ReceiversJson = inject()
    elevation: ElevationAsset = inject()
    crs: WorkingCrsAsset = inject()
    aoi: AreaOfInterestAsset = inject()
    chunksize: ChunksizeAsset = inject()
    area: AreaAsset = inject()

    @override
    def make(self):
        log.debug("Preparing receivers")
        receiver_groups = {}

        if self.config.d.input.normal is not None:
            receiver_groups['normal'] = get_receivers(
                asset=self.receivers,
                elevation=self.elevation.d,
                working_crs=self.crs.d,
                normal_conf=self.config.d.input.normal,
                aoi=self.aoi.d,
                chunksize=self.chunksize.d
            )

        if self.config.d.input.grid is not None:
            receiver_groups['grid'] = get_grid_receivers(
                area=self.area.d,
                elevation=self.elevation.d,
                grid_conf=self.config.d.input.grid,
                chunksize=self.chunksize.d,
                working_crs=self.crs.d,
                aoi=self.aoi.d
            )

        return ReceiverGroupsAsset(receiver_groups)



def get_receivers(asset: assets.ReceiversJson, elevation: xr.DataArray, working_crs: pyproj.CRS, normal_conf: NormalReceivers, aoi: pyproj.aoi.AreaOfInterest, chunksize: Chunksize) -> xr.Dataset:
    df = (
        pd.DataFrame(asset.data)
        .drop(columns=["x", "y"], errors='ignore')
        .rename(columns=dict(id='receiver'))
        .set_index('receiver')
    )

    # convert to xarray
    ds = xr.Dataset.from_dataframe(df)
    ds = _as_fixed_str(
        ds,
        [
            'receiver',
            'street',
            'housenumber',
            'city',
            'postcode',
            'country',
            'noise_classification'
        ],
        errors='ignore'
    )
    ds['position_lonlat'] = xr.concat([ds['longitude'], ds['latitude']], dim='spatial')
    ds = ds.assign_coords(dict(spatial=['x', 'y'])).drop_vars(['longitude', 'latitude'])

    # Create variables if they don't exist yet
    if 'elevation_m' not in ds:
        ds['elevation_m'] = 'receiver', np.full(ds.sizes['receiver'], np.nan)
    if 'height_m' not in ds:
        ds['height_m'] = 'receiver', np.full(ds.sizes['receiver'], np.nan)

    # Transform to working CRS
    t = pyproj.Transformer.from_crs(CRS.WGS84, working_crs, always_xy=True, area_of_interest=aoi)
    x, y = t.transform(*ds['position_lonlat'].sel(spatial=['x', 'y']).values)
    ds = (
        ds
        .assign(position=(('spatial', 'receiver'), [x, y]))
        .chunk(receiver=chunksize._1d)
    )

    # Fill elevation with real data if not specified
    real_elevations = (
        elevation
        .interp(
            x=ds['position'].sel(spatial='x'),
            y=ds['position'].sel(spatial='y')
        )
        .drop_vars(['x', 'y', 'spatial_ref'], errors='ignore')
    )
    ds['elevation_m'] = ds['elevation_m'].fillna(real_elevations)

    # Use default height if not specified
    ds['height_m'] = ds['height_m'].fillna(normal_conf.default_height)

    # add Z coordinate to position
    position_z = ds['elevation_m'] + ds['height_m']
    ds = ds.reindex(spatial=['x', 'y', 'z']).chunk(spatial=-1)
    ds['position'].loc[dict(spatial='z')] = position_z

    return ds


def get_grid_receivers(area: Area, elevation: xr.DataArray, grid_conf: GridReceivers, chunksize: Chunksize, working_crs: pyproj.CRS, aoi: pyproj.aoi.AreaOfInterest) -> xr.Dataset:
    x_min, y_min = area.botleft
    x_max, y_max = area.topright

    # Determine locations. Explicitly set bounds to avoid floating point inaccuracy.
    num_x = int((x_max - x_min) / grid_conf.spacing) + 1
    x = np.linspace(x_min, x_max, num_x)
    x[0], x[-1] = x_min, x_max

    num_y = int((y_max - y_min) / grid_conf.spacing) + 1
    y = np.linspace(y_min, y_max, num_y)
    y[0], y[-1] = y_min, y_max

    # Create 2D grid data with dimensions (x, y)
    grid_x, grid_y = da_creation.meshgrid(
        da_core.from_array(x, chunks=cast(Any, chunksize._2d)),
        da_core.from_array(y, chunks=cast(Any, chunksize._2d)),
        indexing='ij'
    )
    if grid_conf.constant_elevation is not None:
        grid_elevation = da_wrap.full((num_x, num_y), grid_conf.constant_elevation, chunks=chunksize._2d)
    else:
        grid_elevation = elevation.interp(x=x, y=y).transpose('x', 'y').data
    grid_height = da_wrap.full((num_x, num_y), grid_conf.height, chunks=chunksize._2d)
    grid_z = grid_elevation + grid_height
    grid = da_core.stack([grid_x, grid_y, grid_z], axis=-1).rechunk({-1: -1})  # do not chunk along spatial dimension

    # Flatten grid data and construct Dataset
    grid_receivers = xr.Dataset(
        data_vars=dict(
            position=(['x', 'y', 'spatial'], grid),
            height_m=(['x', 'y'], grid_height),
            elevation_m=(['x', 'y'], grid_elevation),
        ),
        coords=dict(spatial=['x', 'y', 'z'])
    ).stack(receiver=('x', 'y'), create_index=False)


    # Lazily transform coordinates to lon,lat
    t = pyproj.Transformer.from_crs(working_crs, CRS.WGS84, always_xy=True, area_of_interest=aoi)

    def transform_mapblock(pos: xr.DataArray):
        return cast(xr.DataArray, xr.apply_ufunc(
            lambda v: np.stack(t.transform(*v), axis=0),
            pos.transpose('spatial', ...).sel(spatial=list('xyz')),
            input_core_dims=[['spatial', 'receiver']],
            output_core_dims=[['spatial', 'receiver']],
        ))

    grid_receivers['position_lonlat'] = xr.map_blocks(
        transform_mapblock,
        grid_receivers['position'],
        template=grid_receivers['position'].transpose('spatial', ...)
    )

    # print(grid_receivers['position'].chunks)
    # print(grid_receivers['position_lonlat'].chunks)

    return grid_receivers
