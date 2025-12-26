import logging
from typing import cast

import pyproj
import rioxarray
from download_copernicus_fabdem import get_elevation
from rioxarray.raster_array import RasterArray

lon = 7.717416
lat = 53.002227

utm_x = 413931.72
utm_y = 5873287.69

buffer = 2500

logging.basicConfig(level=logging.INFO)


crs = pyproj.CRS('EPSG:25832')

elevation = get_elevation(
    xmin=utm_x-buffer,
    xmax=utm_x+buffer,
    ymin=utm_y-buffer,
    ymax=utm_y+buffer,
    crs=crs,
    folder='cache'
)
print(elevation)
cast(RasterArray, elevation.rio).write_crs(crs, inplace=True)
print(elevation)
elevation.rio.to_raster("output.tif")
