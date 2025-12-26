from typing import Any, cast

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt


config = Config.load().data

fig, ax = plt.subplots()
crs = config.computation.crs_name

# plot background map
ctx.add_basemap(
    ax,
    crs=crs,
    source=str(config.input.tilesfile),
    reset_extent=False,
    alpha=1
)

# load isolines from shapefile
df = cast(gpd.GeoDataFrame, gpd.read_file(r'D:/contour_lines.shp')).to_crs(config.computation.crs_name)
df = df.to_crs(config.computation.crs_name)
df.plot(ax=ax)
plt.show()
