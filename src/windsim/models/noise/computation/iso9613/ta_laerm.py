from __future__ import annotations

import numpy as np
import xarray as xr


def L_Aeq_k_j(ds: xr.Dataset, sound_power_levels: xr.DataArray, a_weighting: xr.DataArray) -> xr.Dataset:
    """Just like L_AT_DW, except that turbines are not summed up yet"""
    ds = ds.copy()
    ds['L_Aeq_k_j'] = (
        10 * np.log10(
            (10 ** (0.1 * (
                sound_power_levels - ds['A_total'] + a_weighting
            )))
            .sum(dim='frequency')
        )
    )
    return ds


def L_Aeq_j(ds: xr.Dataset, exposure_times: xr.DataArray, timeslice_durations: xr.DataArray) -> xr.Dataset:
    ds = ds.copy()
    ds['L_Aeq_j'] = (
        10 * np.log10(
            (exposure_times * 10 ** (0.1 * ds['L_Aeq_k_j']))
            .sum(dim='turbine')
            / timeslice_durations
        )
    )
    return ds


def L_r(ds: xr.Dataset, timeslice_durations: xr.DataArray) -> xr.Dataset:
    ds = ds.copy()
    ds['T_r'] = timeslice_durations.sum()
    ds['L_r'] = (
        10 * np.log10(
            (timeslice_durations * 10 ** (0.1 * ds['L_Aeq_j']))
            .sum(dim='timeslice')
            / ds['T_r']
        )
    )
    return ds
