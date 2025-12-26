from __future__ import annotations

from typing import cast

import numpy as np
import xarray as xr
from scipy.spatial import distance_matrix
from trimesh import Trimesh

from . import barrier


def distance(ds: xr.Dataset, turbines: xr.Dataset, receivers: xr.Dataset) -> xr.Dataset:
    """Computes distances between each turbine and receiver."""
    ds = ds.copy()

    def distance_mapblock(a: xr.DataArray, b: xr.DataArray):
        return cast(xr.DataArray, xr.apply_ufunc(
            distance_matrix, a, b,
            input_core_dims=[['turbine', 'spatial'], ['receiver', 'spatial']],
            output_core_dims=[['turbine', 'receiver']],
        ))

    ds['distance'] = xr.map_blocks(
        distance_mapblock,
        turbines['position'],
        args=[receivers['position']]
    )

    return ds

# TODO: consider ground profile for projected distance
def distance_proj(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds['distance_proj'] = ds['distance']
    return ds

def avg_height(ds: xr.Dataset, turbines: xr.Dataset, receivers: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds['avg_height'] = (turbines['hub_height_m'] + receivers['height_m']) / 2
    return ds

def A_atm(ds: xr.Dataset, alpha: xr.DataArray) -> xr.Dataset:
    ds = ds.copy()
    ds['A_atm'] = alpha * ds['distance'] / 1000
    return ds


def is_blocked(
    ds: xr.Dataset,
    turbines: xr.Dataset,
    receivers: xr.Dataset,
    barriers: Trimesh
):
    ds = ds.copy()

    ds['is_blocked'] = xr.map_blocks(
        barrier.is_blocked.is_blocked_mapblock,
        turbines['position'],
        args=[receivers['position'], barriers]
    )

    return ds


def D_Z_top(
    ds: xr.Dataset,
    turbines: xr.Dataset,
    receivers: xr.Dataset,
    barriers: Trimesh,
    frequencies: xr.DataArray
) -> xr.Dataset:
    ds = ds.copy()

    ds['D_Z_top'] = xr.map_blocks(
        barrier.D_Z_top.D_Z_top_mapblock,
        turbines['position'],
        args=[receivers['position'], frequencies, barriers]
    )

    return ds


def D_Z_sides(
    ds: xr.Dataset,
    turbines: xr.Dataset,
    receivers: xr.Dataset,
    barriers: Trimesh,
    frequencies: xr.DataArray
) -> xr.Dataset:
    ds = ds.copy()

    ds['D_Z_sides'] = xr.map_blocks(
        barrier.D_Z_sides.D_Z_sides_mapblock,
        turbines['position'],
        args=[receivers['position'], frequencies, barriers]
    )

    return ds


def A_bar_top(ds: xr.Dataset) -> xr.Dataset:
    """Requires `D_Z_top`."""
    ds = ds.copy()

    ds['A_bar_top'] = xr.where(
        ds['A_gr'] > 0,
        (ds['D_Z_top'] - ds['A_gr']).clip(min=0),
        ds['D_Z_top'].clip(min=0)
    )

    return ds

def A_bar_sides(ds: xr.Dataset) -> xr.Dataset:
    """Requires `D_Z_sides`."""
    ds = ds.copy()

    ds['A_bar_sides'] = ds['D_Z_sides'].clip(min=0)

    return ds


# def A_bar_side1(ds: xr.Dataset) -> xr.Dataset:
#     """Requires `D_Z_side1`."""
#     ds = ds.copy()

#     ds['A_bar_side1'] = ds['D_Z_side1'].clip(min=0)

#     return ds

# def A_bar_side2(ds: xr.Dataset) -> xr.Dataset:
#     """Requires `D_Z_side2`."""
#     ds = ds.copy()

#     ds['A_bar_side2'] = ds['D_Z_side2'].clip(min=0)

#     return ds


def A_bar(ds: xr.Dataset) -> xr.Dataset:
    """Requires `A_bar_top`, `A_bar_sides`."""
    ds = ds.copy()

    ds['A_bar'] = (
        -10 * np.log10(
            10 ** (-0.1 * ds['A_bar_top'])
            + 10 ** (-0.1 * ds['A_bar_sides'].sel(side='left'))
            + 10 ** (-0.1 * ds['A_bar_sides'].sel(side='right'))
        )
    ).clip(min=0)


    # for each sound-receiver combination, determine barriers that intersect direct line from source to receiver
    # of those, filter out the ones that are too short
    # each barrier is represented as shapely polygon


    # find planes E_V and E_L.
    # E_V is perpendicular to xy plane:
    #   Normal is cross product between source-receiver line and the xy normal (0,0,1)
    #   Origin is the source
    # E_L is perpendicular to E_V:
    #   Normal is cross product between source-receiver line and normal of E_V
    #   Origin is the source

    # Question: Do we want to "cut" polygons, or do we want to either consider them fully or not at all?
    # Or does this mean we would be show an edge where there really is none?

    return ds

def A_div(ds: xr.Dataset) -> xr.Dataset:
    d_0 = 1  # reference value
    ds = ds.copy()
    ds['A_div'] = 20 * np.log10(ds['distance'] / d_0) + 11
    return ds

def A_misc(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds['A_misc'] = 0
    return ds

def A_gr(ds: xr.Dataset, turbines: xr.Dataset, receivers: xr.Dataset, coarseness: xr.Dataset) -> xr.Dataset:
    def a(h):
        return (
            1.5
            + 3.0 * np.exp(-0.12 * (h-5)**2) * (1 - np.exp(-ds['distance_proj']/50))
            + 5.7 * np.exp(-0.009 * h**2) * (1 - np.exp(-2.8 * 10**(-6) * ds['distance_proj']**2))
        )
    
    def b(h):
        return (
            1.5
            + 8.6 * np.exp(-0.09 * h**2) * (1 - np.exp(-ds['distance_proj']/50))
        )
    
    def c(h):
        return (
            1.5
            + 14.0 * np.exp(-0.46 * h**2) * (1 - np.exp(-ds['distance_proj']/50))
        )

    def d(h):
        return (
            1.5
            + 5.0 * np.exp(-0.9 * h**2) * (1 - np.exp(-ds['distance_proj']/50))
        )
    
    ds = ds.copy()

    G_s = coarseness['G_s']
    ds['A_s'] = xr.concat([
        xr.DataArray(-1.5),
        -1.5 + G_s * a(turbines['hub_height_m']),
        1.5 + G_s * b(turbines['hub_height_m']),
        -1.5 + G_s * c(turbines['hub_height_m']),
        -1.5 + G_s * d(turbines['hub_height_m']),
        -1.5 * (1 - G_s),
        -1.5 * (1 - G_s),
        -1.5 * (1 - G_s)
    ], dim="frequency")

    G_r = coarseness['G_r']
    ds['A_r'] = xr.concat([
        xr.DataArray(-1.5),
        -1.5 + G_r * a(receivers['height_m']),
        1.5 + G_r * b(receivers['height_m']),
        -1.5 + G_r * c(receivers['height_m']),
        -1.5 + G_r * d(receivers['height_m']),
        -1.5 * (1 - G_r),
        -1.5 * (1 - G_r),
        -1.5 * (1 - G_r)
    ], dim="frequency")

    q = xr.where(
        ds['distance_proj'] > 30 * (turbines['hub_height_m'] + receivers['height_m']),
        1 - (30 * (turbines['hub_height_m'] + receivers['height_m'])) / ds['distance_proj'],
        0
    )

    G_m = coarseness['G_m']
    ds['A_m'] = xr.concat([
        -3 * q,
        -3 * q * (1 - G_m),
        -3 * q * (1 - G_m),
        -3 * q * (1 - G_m),
        -3 * q * (1 - G_m),
        -3 * q * (1 - G_m),
        -3 * q * (1 - G_m),
        -3 * q * (1 - G_m)
    ], dim="frequency")

    ds['A_gr'] = ds['A_s'] + ds['A_r'] + ds['A_m']

    return ds


def A_gr_alternative(ds: xr.Dataset, turbines: xr.Dataset, receivers: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()

    ds['A_gr'] = (
        4.8 - (2 * ds['avg_height']/ds['distance']) * (17 + 300/ds['distance'])
    ).clip(min=0)

    ds['D_omega'] = (
        10 * np.log10(
            1
            + (
                ds['distance_proj']**2 + (turbines['hub_height_m'] - receivers['height_m'])**2
            )
            / (
                ds['distance_proj']**2 + (turbines['hub_height_m'] + receivers['height_m'])**2
            )
        )
    )

    return ds

def A_total(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    ds['A_total'] = (
        ds['A_div']
        + ds['A_gr']
        + ds['A_bar']
        + ds['A_misc']
        + ds['A_atm']
    )
    return ds

def D_I(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(D_I=0)

def D_C(ds: xr.Dataset) -> xr.Dataset:
    if 'D_omega' in ds:
        return ds.assign(D_C=ds['D_I'] + ds['D_omega'])
    return ds.assign(D_C=ds['D_I'])

def L_fT_DW(ds: xr.Dataset, sound_power_levels: xr.DataArray) -> xr.Dataset:
    return ds.assign(L_fT_DW=sound_power_levels + ds['D_C'] - ds['A_total'])

def L_AT_DW(ds: xr.Dataset, sound_power_levels: xr.DataArray, a_weighting: xr.DataArray) -> xr.Dataset:
    ds = ds.copy()
    ds['L_AT_DW'] = (
        10 * np.log10(
            (10 ** (0.1 * (
                sound_power_levels - ds['A_total'] + a_weighting
            )))
            .sum(dim=['frequency', 'turbine'])
        )
    )
    return ds

def C_met(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(C_met=0)

def L_AT_LT(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(L_AT_LT=ds['L_AT_DW'] - ds['C_met'])
