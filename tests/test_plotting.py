from types import SimpleNamespace

import matplotlib

matplotlib.use('Agg')

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyproj
import pytest
import xarray as xr

from windsim.models.noise.config.sections.output import OutputMapSection
from windsim.models.noise.input.area import Area
from windsim.models.noise.output import plotting


@pytest.mark.parametrize('with_tiles', [False, True])
@pytest.mark.parametrize('use_computation_crs', [False, True])
@pytest.mark.parametrize('crs_kind', ['utm', 'local'])
def test_noise_map_background_is_optional(
    tmp_path, monkeypatch, with_tiles, use_computation_crs, crs_kind,
):
    raw = {
        'indicate_input_area': True,
        'individual_colors': True,
        'use_computation_crs': use_computation_crs,
        'contour_levels': [20, 30, 40, 50],
        'contour_fill_visibility': 0.65,
        'contour_line_visibility': 1,
        'add_contour_labels': False,
        'add_receiver_labels': True,
    }
    if with_tiles:
        raw['tiles'] = {'api_key': 'test-key', 'zoom': 1}
    config = SimpleNamespace(output=SimpleNamespace(map=OutputMapSection(raw)))
    tile_requests = []
    backgrounds = []

    class OfflineTiles:
        crs = ccrs.Mercator.GOOGLE

        def image_for_domain(self, domain, zoom):
            backgrounds.append((self, zoom))
            xmin, ymin, xmax, ymax = domain.bounds
            image = np.full((16, 16, 3), 235, dtype=np.uint8)
            return image, (xmin, xmax, ymin, ymax), 'lower'

    def make_tiles(**kwargs):
        tile_requests.append(kwargs)
        return OfflineTiles()

    monkeypatch.setattr(plotting.cimgt, 'StadiaMapsTiles', make_tiles)

    if crs_kind == 'utm':
        working_crs = pyproj.CRS.from_epsg(32632)
        x0, y0 = 500000, 6200000
    else:
        working_crs = pyproj.CRS(
            proj='tmerc', lat_0=56.31, lon_0=8.59, datum='WGS84', units='m',
        )
        x0, y0 = 0, 0
    x = np.linspace(x0, x0 + 1000, 5)
    y = np.linspace(y0, y0 + 1000, 5)
    grid = xr.Dataset(
        {'L_AT_LT': (('x', 'y'), np.linspace(20, 50, 25).reshape(5, 5))},
        coords={'x': x, 'y': y},
    )
    receivers = xr.Dataset(
        {
            'position': (('receiver', 'spatial'), [[x[2], y[2], 5]]),
            'L_AT_LT': ('receiver', [35.0]),
        },
        coords={'name': ('receiver', ['R01']), 'spatial': ['x', 'y', 'z']},
    )

    try:
        plotting.plot(
            plot_variable='L_AT_LT',
            area=Area((x[0], y[0]), (x[-1], y[-1])),
            working_crs=working_crs,
            grid_restructured=grid,
            normal_restructured=receivers,
            folder=tmp_path,
            config=config,
        )
        expected_crs = (
            ccrs.Projection(working_crs)
            if not with_tiles or use_computation_crs
            else ccrs.Mercator.GOOGLE
        )
        figure = plt.gcf()
        axes = figure.axes[0]
        assert axes.projection == expected_crs
        assert len(tile_requests) == len(backgrounds) == int(with_tiles)
        if with_tiles:
            assert tile_requests[0]['apikey'] == 'test-key'
            assert backgrounds[0][1] == 1
        with Image.open(tmp_path / 'output.png') as image:
            assert image.width > 100 and image.height > 100
            pixels = np.asarray(image.convert('RGB')).astype(int)
            transform = pyproj.Transformer.from_crs(
                working_crs, pyproj.CRS(axes.projection), always_xy=True,
            )
            colors = []
            for index in [1, 3]:
                point = transform.transform(x[index], y[index])
                u, v = axes.transData.transform(point)
                column = int(u / figure.bbox.width * image.width)
                row = int((1 - v / figure.bbox.height) * image.height)
                color = pixels[row, column]
                assert np.ptp(color) > 20
                colors.append(color)
            assert np.max(np.abs(colors[0] - colors[1])) > 20
    finally:
        plt.close('all')
