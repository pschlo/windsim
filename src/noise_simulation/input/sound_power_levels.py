import logging
from typing import override, cast
import xarray as xr

from planner import Asset, Recipe, DataAsset, inject
from .full_turbines import FullTurbinesAsset
from .turbine_types import MyTurbineTypesAsset
from .timeslice_durations import TimesliceDurationsAsset
from ..config import ConfigAsset


log = logging.getLogger(__name__)


class SoundPowerLevelsAsset(DataAsset[xr.DataArray]):
    pass


class SoundPowerLevelsRecipe(Recipe[SoundPowerLevelsAsset]):
    _makes = SoundPowerLevelsAsset
    
    config: ConfigAsset = inject()
    turbines: FullTurbinesAsset = inject()
    turbine_types: MyTurbineTypesAsset = inject()
    timeslices: TimesliceDurationsAsset = inject()

    @override
    def make(self):
        log.debug("  Preparing sound power levels")

        # Match the sound power levels to each turbine based on its model
        da = (
            self.turbine_types.d['sound_power_level_db']
            .sel(model=self.turbines.d['turbine_type'])
            .drop_vars('model')
        )

        # add timeslice dimension
        da = da.expand_dims(timeslice=self.timeslices.d.coords['timeslice']).copy()

        # Example: turbine runs slightly quieter during the night
        da.loc[dict(timeslice='night')] -= 3

        sound_power_levels = da

        # DEBUG
        sound_power_levels = sound_power_levels.sel(timeslice=['day'])
        return SoundPowerLevelsAsset(sound_power_levels)
