from __future__ import annotations

import xarray as xr


def A_gr_interim(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(A_gr=-3)


def D_C_interim(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(D_C=0)


def C_met_interim(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(C_met=0)
