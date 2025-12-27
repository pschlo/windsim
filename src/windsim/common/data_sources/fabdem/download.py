import logging
import math
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pyproj.aoi
import requests as req
import rioxarray
import rioxarray.merge
import rioxarray.raster_array
import xarray as xr
from geopandas import GeoDataFrame, GeoSeries
from rasterio.enums import Resampling
from rioxarray.raster_array import RasterArray


log = logging.getLogger(__name__)


BASE_URL = 'https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn'
VERSION = 'v1-2'
RESOLUTION_DEG = 1 / (60*60)  # 1 arcsecond


def parse_tilename(name: str):
    """Extracts longitude and latitude from a given tile name."""
    match = re.match(r'([NS])(\d+)([EW])(\d+)', name)
    assert match is not None
    lat_dir, lat, lon_dir, lon = match.groups()
    lat_min = int(lat) * (-1 if lat_dir == 'S' else 1)
    lon_min = int(lon) * (-1 if lon_dir == 'W' else 1)
    lat_max = lat_min + 1
    lon_max = lon_min + 1
    return lon_min, lat_min, lon_max, lat_max


def fetch_overview():
    overview_url = f"{BASE_URL}/FABDEM_{VERSION}_tiles.geojson"
    r = req.get(overview_url)
    r.raise_for_status()
    df = (
        GeoDataFrame.from_features(r.json())
        .drop(columns='geometry')
        .rename(columns=dict(tile_name='name', file_name='filename', zipfile_name='zipname'))
        .assign(lonlat=lambda df: df['name'].apply(parse_tilename))
        .set_index('lonlat')
    )
    return df


def determine_required_tiles(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> pd.DataFrame:
    # Determine required tiles
    tuples: list[tuple[int,int,int,int]] = []
    for lon in range(math.floor(lon_min), math.ceil(lon_max)):
        for lat in range(math.floor(lat_min), math.ceil(lat_max)):
            tuples.append((lon, lat, lon + 1, lat + 1))
    x = pd.DataFrame(dict(lonlat=tuples)).set_index(['lonlat'])
    return x


def format_bytes(size: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = int(math.log(size, 1000))
    return f"{round(size / 1000**i, 2)} {units[i]}"


class Progress:
    min_interval_s = 5

    def __init__(self, total_size: int) -> None:
        self.total_size = total_size
        self.cum_received = 0
        self.wait_until = 0

    def log(self):
        _a, _b = format_bytes(self.cum_received), format_bytes(self.total_size)
        _percent = round((self.cum_received / self.total_size) * 100)
        log.info(f"  Progress: {_a} / {_b} ({_percent}%)")

    def callback(self, received: int):
        self.cum_received += received
        if time.time() >= self.wait_until:
            self.wait_until = time.time() + self.min_interval_s
            self.log()

    def finish(self):
        self.log()


def fetch_tiles(tiles_df, folder: Path | str):
    folder = Path(folder).resolve()
    folder.mkdir(exist_ok=True)

    zipfiles = tiles_df['zipname'].unique()

    # Determine full download size
    sizes = {f: int(req.head(f"{BASE_URL}/{f}").headers['content-length']) for f in zipfiles}
    _s = format_bytes(sum(sizes.values()))
    log.info(f"Fetching {len(zipfiles)} tile collections ({_s})")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, zipfile in enumerate(zipfiles):
            _s = format_bytes(sizes[zipfile])
            log.info(f'Fetching tile collection {i+1} of {len(zipfiles)}: {zipfile} ({_s})')

            # Download
            download_url = f"{BASE_URL}/{zipfile}"
            r = req.get(download_url, stream=True)
            r.raise_for_status()
            progress = Progress(sizes[zipfile])
            with (tmpdir / zipfile).open('wb') as f:
                for chunk in r.iter_content(chunk_size=10 * 1000**2):
                    f.write(chunk)
                    progress.callback(len(chunk))
            progress.finish()

            # Unzip requested files
            with ZipFile(tmpdir / zipfile, 'r') as zip_ref:
                zip_df = pd.DataFrame(
                    [(name, parse_tilename(name)) for name in zip_ref.namelist()],
                    columns=['filename', 'lonlat']
                ).set_index('lonlat')
                missing_zip_df = zip_df.loc[zip_df.index.intersection(tiles_df.index)]
                zip_ref.extractall(folder, missing_zip_df['filename'])


def get_existing_tiles(folder: Path | str):
    folder = Path(folder).resolve()
    existing_df = pd.DataFrame(
        [(f.name, parse_tilename(f.name)) for f in folder.iterdir()],
        columns=['filename', 'lonlat']
    ).set_index('lonlat')
    return existing_df


def prepare_area_tiles(lon_min: float, lat_min: float, lon_max: float, lat_max: float, folder: Path|str):
    """Ensures all tiles required for the given area are present on disk. Downloads tiles if required.
    Returns a Dataframe containing the required local tiles."""
    folder = Path(folder).resolve()

    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError('Invalid area bounds')

    log.info("Preparing elevation tiles")
    folder.mkdir(exist_ok=True)

    # Determine tiles without network communication
    required_df = determine_required_tiles(lon_min, lat_min, lon_max, lat_max)
    existing_df = get_existing_tiles(folder)
    missing_df = required_df.loc[required_df.index.difference(existing_df.index)]
    if len(missing_df) == 0:
        log.info(f"All of {len(required_df)} required elevation tiles exist on disk")
    else:
        log.info(f"{len(missing_df)} of {len(required_df)} elevation tiles are missing")
        # Fetch overview to determine zip collections and download missing tiles
        overview_df = fetch_overview()
        if not missing_df.index.isin(overview_df.index).all():
            raise RuntimeError(
                'Some elevation tiles are not available for the selected area. Please choose a different area that primarily includes land-based regions.'
            )
        fetch_tiles(overview_df.loc[missing_df.index], folder)

    existing_df = get_existing_tiles(folder)
    return existing_df.loc[required_df.index]


def clip_exact(array: xr.DataArray, xmin, ymin, xmax, ymax) -> xr.DataArray:
    """Clip and interpolate such that bounds are exactly the area."""
    # Slicing selects all values inside the range (inclusive)
    x_new = array['x'].sel(x=slice(xmin, xmax)).values
    x_new = np.unique(np.hstack([xmin, x_new, xmax]))
    y_new = array['y'].sel(y=slice(ymin, ymax)).values
    y_new = np.unique(np.hstack([ymin, y_new, ymax]))
    return array.interp(x=x_new, y=y_new, assume_sorted=True)

def clip_contains(array: xr.DataArray, xmin, ymin, xmax, ymax) -> xr.DataArray:
    """Clip such that area bounds are contained within data bounds."""
    clip_xmin = array['x'].sel(x=xmin, method='ffill').item()
    clip_ymin = array['y'].sel(y=ymin, method='ffill').item()
    clip_xmax = array['x'].sel(x=xmax, method='bfill').item()
    clip_ymax = array['y'].sel(y=ymax, method='bfill').item()
    # Slicing selects all values inside the range (inclusive)
    return array.sel(x=slice(clip_xmin, clip_xmax), y=slice(clip_ymin, clip_ymax))


def get_elevation_lonlat(lon_min: float, lat_min: float, lon_max: float, lat_max: float, folder: Path|str, exact_bounds: bool = False) -> xr.DataArray:
    folder = Path(folder)
    area = lon_min, lat_min, lon_max, lat_max
    tiles_df = prepare_area_tiles(*area, folder=folder)

    arrays: list[xr.DataArray] = []
    for filename in tiles_df['filename']:
        # Load raster and replace nodata value (often somethink like -9999.0) with nan to ensure interpolations work.
        raster = rioxarray.open_rasterio(folder / filename, mask_and_scale=True)
        assert isinstance(raster, xr.DataArray)
        assert np.isnan(raster.rio.nodata)
        raster = raster.squeeze('band').drop_vars('band')
        arrays.append(raster)

    array = rioxarray.merge.merge_arrays(arrays)
    array = array.sortby(['x', 'y'])
    array = clip_exact(array, *area) if exact_bounds else clip_contains(array, *area)

    if np.isnan(array).any():
        raise ValueError('Data missing in elevation tiles')

    return array


def get_elevation(xmin: float, ymin: float, xmax: float, ymax: float, crs: pyproj.CRS, folder: Path|str, exact_bounds: bool = True) -> xr.DataArray:
    area = xmin, ymin, xmax, ymax

    # Convert area to WGS84 lon/lat
    t = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lon_min, lat_min, lon_max, lat_max = t.transform_bounds(*area)

    # During reprojection, elevation points are resampled. This may cause them to lie just outside of the specified area.
    # Add small buffer to make area slightly larger.
    BUFFER = 3 * RESOLUTION_DEG
    lon_min -= BUFFER
    lat_min -= BUFFER
    lon_max += BUFFER
    lat_max += BUFFER

    # Get elevation and reproject and clip to CRS
    array = get_elevation_lonlat(lon_min, lat_min, lon_max, lat_max, folder=folder, exact_bounds=False)
    array = cast(RasterArray, array.rio).reproject(crs, resampling=Resampling.bilinear)
    array = array.sortby(['x', 'y'])
    array = clip_exact(array, *area) if exact_bounds else clip_contains(array, *area)

    if np.isnan(array).any():
        raise ValueError('Data missing in elevation tiles')

    return array




def playground():
    lon_min, lat_min = 5.869837267732521, 50.7572449215288
    lon_max, lat_max = 6.137564149788657, 50.93728505518783

    # area is given as UTM rectangle
    xmin, ymin = 282266.4045493543, 5631342.960495994
    xmax, ymax = 293588.59951231437, 5640951.92851497

    folder = Path(r'D:\fabdem_elevation')
    elevation = get_elevation(xmin, ymin, xmax, ymax, crs=pyproj.CRS('epsg:25832'), folder=folder)
    # elevation = get_elevation_lonlat(lon_min, lat_min, lon_max, lat_max, folder=folder)


    w, h = (xmax-xmin), (ymax-ymin)
    p1 = xmin + w/2, ymin + h/2
    p2 = xmin+w/2, ymin+h/2 + 0.00001
    p3 = xmin+w/2, ymin
    # print(p1)
    # print(p2)
    # print(p3)
    p4 = xmin, ymin
    p5 = xmin, ymax
    ps = np.array([p1, p2, p3, p4, p5])
    interp = elevation.interp(
        x=xr.DataArray(ps[:, 0], dims="points"),
        y=xr.DataArray(ps[:, 1], dims="points")
    )

    print(interp)


    # DEBUG
    # box = shapely.box(xmin, ymin, xmax, ymax)
    # plot_polygon(box, add_points=False, edgecolor='red', linewidth=1, zorder=10)
    # x, y = np.meshgrid(elevation['x'].values, elevation['y'].values)
    # plt.scatter(x, y, color='red', s=1, zorder=10)

    elevation.plot.pcolormesh(x='x', y='y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
