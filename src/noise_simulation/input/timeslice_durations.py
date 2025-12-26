import logging
from typing import override, cast
import xarray as xr

from planner import Asset, Recipe, DataAsset, inject, assets
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class TimesliceDurationsAsset(DataAsset[xr.DataArray]):
    pass


class TimesliceDurationsRecipe(Recipe[TimesliceDurationsAsset]):
    _makes = TimesliceDurationsAsset
    
    config: ConfigAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing timeslice durations")

        # Example: 16h day, 12h night
        ids = ['day', 'night']
        durations = [16, 8]
        timeslice_durations = xr.DataArray(
            durations,
            coords={'timeslice': ids}
        )

        # DEBUG
        # timeslice_durations = timeslice_durations.sel(timeslice=['day'])
        return TimesliceDurationsAsset(timeslice_durations)
