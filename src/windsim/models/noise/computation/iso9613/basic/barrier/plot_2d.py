from collections.abc import Iterable

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.plotting
from matplotlib_scalebar.scalebar import ScaleBar


def plot_2d(s, r, polygons: Iterable[shapely.Polygon], path: shapely.LineString | None = None, path2: shapely.LineString|None = None):
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # disable all ticks and labels
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False
    )

    for p in polygons:
        shapely.plotting.plot_polygon(p, add_points=False, color='dimgrey', ax=ax)
    ax.plot(*s, 'go', markersize=10)
    ax.plot(*r, 'ro', markersize=10)
    if path is not None:
        shapely.plotting.plot_line(path, add_points=False, linewidth=1, color='blue', ax=ax)
    if path2 is not None:
        shapely.plotting.plot_line(path2, add_points=False, linewidth=1, color='blue', ax=ax)

    # ground = matplotlib.patches.Rectangle((-500, -50), 1500, -200, hatch='/', fill=False, edgecolor='lightgray')
    # ax.add_patch(ground)

    ax.add_artist(ScaleBar(1, 'm'))

    plt.show()
