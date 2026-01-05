from __future__ import annotations

from pathlib import Path
from typing import Any, Type, cast

import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.mpl
import cartopy.mpl.geoaxes
import cartopy.vector_transform
import geopandas as gpd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import shapely
import shapely.ops
import xarray as xr
from matplotlib.artist import Artist
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection
from matplotlib.transforms import Affine2D
from matplotlib_scalebar.scalebar import ScaleBar
from shapely import LinearRing, MultiPolygon, Point, Polygon
from shapely.plotting import plot_polygon

from windsim.coordinate_reference_systems import CRS
from windsim.models.noise.verification.verify_iso_lines import compare_contours

from ..config import ConfigData
from ..input.area import Area
from .framed_scalebar import FramedScaleBar


LABEL_BBOX = dict(boxstyle='round,pad=0.2', fc='white', alpha=1, edgecolor='grey')


def plot(
    plot_variable: str,
    area: Area,
    working_crs: pyproj.CRS,
    grid_restructured: xr.Dataset | None = None,
    features: list[Polygon] | MultiPolygon | None = None,
    turbines: xr.Dataset | None = None,
    normal_restructured: xr.Dataset | None = None,
    elevation: xr.DataArray | None = None,
    *,
    folder: Path,
    config: ConfigData
):
    """Create plot specified by the config file.
    """

    if config.output.map is None:
        raise RuntimeError("Map output must be enabled in config")

    # use latex for rendering
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "Helvetica",
    #     'font.size': 22
    # })

    tileconf = config.output.map.tiles
    tiler = cimgt.StadiaMapsTiles(
        style=tileconf.style,
        resolution='@2x' if tileconf.double_resolution else '',
        apikey=tileconf.api_key)

    # define coordinate reference systems
    working_crs_pyproj = working_crs
    working_crs_cartopy = ccrs.Projection(working_crs)
    if config.output.map.use_computation_crs:
        map_crs_pyproj = working_crs
        map_crs_cartopy = ccrs.Projection(working_crs)
    else:
        map_crs_cartopy = tiler.crs
        map_crs_pyproj = pyproj.CRS.from_user_input(tiler.crs)

    # create cartopy plot
    fig, ax = plt.subplots(subplot_kw=dict(projection=map_crs_cartopy))
    assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxes)

    buffer = config.output.map.buffer
    ax.set_extent([
        area.botleft[0] - buffer,
        area.topright[0] + buffer,
        area.botleft[1] - buffer,
        area.topright[1] + buffer,
    ], crs=working_crs_cartopy)

    ax.add_image(tiler, tileconf.zoom)

    if grid_restructured is not None:
        # DEBUG: add WindPRO contour lines
        # cs_lines = grid_restructured[plot_variable].plot.contour(
        #     ax=ax,
        #     levels=config.output.map.contour_levels,
        #     x='x',
        #     y='y',
        # )
        # compare_contours(cs_lines, r'D:\Noise Verification\Contours radius7km grid1m levels0.25\contours.shp', crs=working_crs_pyproj, num_linepoints=50_000)
        # exit()

        cs_filled = grid_restructured[plot_variable].plot.contourf(
            ax=ax,
            levels=config.output.map.contour_levels,
            x='x',
            y='y',
            alpha=config.output.map.contour_fill_visibility,
            cmap='viridis',
            add_colorbar=True,
            add_labels=False,
            transform=working_crs_cartopy
        )

        # get the colorbar Axes
        cbar_ax = [_ax for _ax in fig.axes if _ax != ax][0]

        cs_lines = grid_restructured[plot_variable].plot.contour(
            ax=ax,
            levels=config.output.map.contour_levels,
            x='x',
            y='y',
            alpha=config.output.map.contour_line_visibility,
            cmap='viridis',
            linewidths=3,
            cbar_ax=cbar_ax,
            cbar_kwargs={'label': 'Equivalent continuous A-weighted sound pressure level (dB(A))'},
            add_colorbar=True,
            add_labels=False,
            transform=working_crs_cartopy
        )

        # add labels to contour lines
        if config.output.map.add_contour_labels:
            labels = ax.clabel(cs_lines, inline=False, fontsize=10, colors='black')
            for label in labels:
                label.set_alpha(1)
                label.set_rotation(0)
                label.set_bbox(LABEL_BBOX)

        # add grid receivers
        # x, y = np.meshgrid(grid_restructured['x'], grid_restructured['y'])
        # ax.scatter(
        #     x=x, y=y, label="Grid Receivers", marker=cast(Any, '.'), s=0.5, color='red', zorder=2
        # )

    # add features
    if features is not None:
        if not isinstance(features, MultiPolygon):
            features = MultiPolygon(features)
        plot_polygon(
            features,
            ax=ax,
            facecolor='gray',
            linewidth=0.5,
            edgecolor='red',
            add_points=False,
            transform=working_crs_cartopy
        )

    # add turbines
    if turbines is not None:
        pos = turbines['position'].transpose('turbine', 'spatial').values

        # dark shadow
        ax.scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            marker=cast(Any, '1'),
            s=375,
            linewidth=4,
            color='black',
            zorder=2,
            transform=working_crs_cartopy
        )

        # green inside
        ax.scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            label="Turbine",
            marker=cast(Any, '1'),
            s=300,
            linewidth=2,
            color='limegreen',
            zorder=2,
            transform=working_crs_cartopy
        )


    # add normal receivers
    if normal_restructured is not None:
        pos = normal_restructured['position'].transpose('receiver', 'spatial').values

        # Either label and color each point individually, or same for each
        opts = dict(marker=cast(Any, '.'), s=200, edgecolor='black', zorder=2, transform=working_crs_cartopy)
        cmap = plt.get_cmap('tab10')
        if config.output.map.individual_colors and len(pos) <= cmap.N:
            _c = 0
            labels = [
                t if t
                else f"IO {(_c := _c + 1)}"
                for t in normal_restructured['name'].values
            ]
            colors = cmap(np.arange(len(pos)))
            for i in range(len(pos)):
                ax.scatter(pos[i, 0], pos[i, 1], label=labels[i], color=colors[i], **opts)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], label='Receiver', color='red', **opts)

        # add result labels
        if config.output.map.add_receiver_labels:
            res = normal_restructured[plot_variable].values
            for i, value in enumerate(res):
                value = round(value, 2)
                ax.annotate(
                    value,
                    pos[i, :2],
                    bbox=LABEL_BBOX,
                    textcoords="offset points",
                    xytext=(4, 8),
                    ha='left',
                    transform=working_crs_cartopy
                )


    # add rectangle to indicate calculated area
    if config.output.map.indicate_input_area:
        rect = patches.Rectangle(
            (area.botleft[0], area.botleft[1]),
            area.topright[0] - area.botleft[0],
            area.topright[1] - area.botleft[1],
            edgecolor='black',
            linewidth=3,
            facecolor='none',
            zorder=2.9,
            transform=working_crs_cartopy
        )
        ax.add_patch(rect)

    # if elevation is not None:
    #     elevation.plot.pcolormesh(ax=ax, x='x', y='y', alpha=0.5, cbar_kwargs={'label': 'Elevation (m)'}, transform=working_crs_cartopy)

    # determine 1 tile unit = x working_crs units
    working_to_tile = pyproj.Transformer.from_crs(working_crs_pyproj, map_crs_pyproj, always_xy=True)
    x1, y1 = (area.botleft[0] + area.topright[0]) / 2, (area.topright[1] + area.botleft[1]) / 2
    delta = 100
    x2, y2 = x1 + delta, y1
    x1_tile, y1_tile = working_to_tile.transform(x1, y1)
    x2_tile, y2_tile = working_to_tile.transform(x2, y2)
    distance_tile = np.linalg.norm([x2_tile - x1_tile, y2_tile - y1_tile])
    working_units_per_tileunit = delta / distance_tile

    # add scalebar
    scalebar = FramedScaleBar(
        working_units_per_tileunit, 'm',
        border_pad=0,
        pad=0.8,
        location='lower right'
    )
    ax.add_artist(scalebar)

    ax.set_aspect('equal')
    ax.set_title("Equivalent continuous A-weighted sound pressure levels (dB(A))", pad=20)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    # See https://stackoverflow.com/a/47116009
    marker_size = 200
    def update_prop(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([marker_size])
    handler_map = {PathCollection: HandlerPathCollection(update_func=update_prop)}
    ax.legend(loc='upper left', framealpha=1, handler_map=handler_map)
    
    # plt.show()
    plt.savefig(folder / 'output.png', dpi=300)
