import logging
from typing import override
import xarray as xr

from planner import Asset, Recipe, DataAsset, inject, assets
from .full_turbines import FullTurbinesAsset
from .timeslice_durations import TimesliceDurationsAsset
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class ExposureTimesAsset(DataAsset[xr.DataArray]):
    pass


class ExposureTimesRecipe(Recipe[ExposureTimesAsset]):
    _makes = ExposureTimesAsset
    
    config: ConfigAsset = inject()
    turbines: FullTurbinesAsset = inject()
    timeslices: TimesliceDurationsAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing exposure times")
        timeslice_durations = self.timeslices.d
        turbines = self.turbines.d

        # Example: every turbine is running all the time,
        # i.e. for every time slice just as long as the duration of the time slice
        # TODO: do not rely on coords being set
        exposure_times = timeslice_durations.expand_dims(turbine=turbines.coords['turbine'])
        return ExposureTimesAsset(exposure_times)
