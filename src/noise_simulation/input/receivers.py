import xarray as xr
import numpy as np
import logging
from typing import override

from planner import Asset, Recipe, DataAsset, inject, assets
from .receiver_groups import ReceiverGroupsAsset


log = logging.getLogger(__name__)


class ReceiversAsset(DataAsset[xr.Dataset]):
    pass


class ReceiversRecipe(Recipe[ReceiversAsset]):
    _makes = ReceiversAsset

    receiver_groups: ReceiverGroupsAsset = inject()

    @override
    def make(self):
        # remove receiver coord and only use relevant datavars
        _receiver_groups_stripped = [
            g.drop_vars('receiver', errors='ignore')[['position', 'position_lonlat', 'elevation_m', 'height_m']] for g in self.receiver_groups.d.values()
        ]

        if self.receiver_groups.d:
            receivers = xr.concat(_receiver_groups_stripped, dim='receiver')
        else:
            # create placeholder empty receiver dataset
            receivers = xr.Dataset(
                data_vars=dict(
                    position=(('receiver', 'spatial'), np.empty((0, 3))),
                    height_m=('receiver', np.empty((0,)))
                )
            )

        return ReceiversAsset(receivers)
