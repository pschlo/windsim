from __future__ import annotations
from collections.abc import Generator, Collection
from typing import override, TypedDict, Unpack, Any
from dataclasses import dataclass, field
from xarray import Dataset
import xarray as xr
from pathlib import Path
from itertools import product
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

from planner import Asset, Recipe, inject, store

from windsim.common.data_sources.cerra_heights import run_pipeline, Area


def get_filename(year: int, month: int) -> str:
    return f"year={year},month={month}.nc"

def parse_filename(filename: str) -> tuple[int, int]:
    match = re.match(r'year=(\d+),month=(\d+).nc', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    year = int(match.group(1))
    month = int(match.group(2))
    return year, month


@dataclass
class CerraStoreAsset(Asset):
    path: Path

    def load(self, year: int | Collection[int] | None = None, month: int | Collection[int] | None = None) -> xr.Dataset:
        area_folder = self.path

        if year is None:
            match_years = None
        else:
            match_years = {year} if isinstance(year, int) else set(year)

        if month is None:
            match_months = None
        else:
            match_months = {month} if isinstance(month, int) else set(month)

        matching_files: set[tuple[int, int, Path]] = set()
        for p in area_folder.iterdir():
            year, month = parse_filename(p.name)
            if not (match_years is None or year in match_years):
                continue
            if not (match_months is None or month in match_months):
                continue
            matching_files.add((year, month, p))
        matching_files_sorted = sorted(matching_files, key=lambda t: (t[0], t[1]))

        ds = xr.open_mfdataset(
            [t[2] for t in matching_files_sorted],
            chunks={}
        )

        # Create hourly time dimension
        ds = ds.stack(_time=("time", "step"), create_index=False)
        ds = ds.assign_coords(_time=ds['valid_time'])
        ds = ds.drop_vars(['time', 'step', 'valid_time']).rename(_time='time')

        return ds

@dataclass(kw_only=True)
class CerraStoreConfAsset(Asset):
    year: int | Collection[int]
    month: int | Collection[int] = range(1,13)
    area: Area | None = None
    api_key: str | None = None


class CerraStoreRecipe(Recipe[CerraStoreAsset]):
    _makes = CerraStoreAsset
    _caps = [
        store.StorageCap(tag="cerra-store", shared=True)
    ]

    storage: store.assets.StorageProvider = inject()
    conf: CerraStoreConfAsset = inject()

    @override
    def make(self):
        years = {self.conf.year} if isinstance(self.conf.year, int) else set(self.conf.year)
        months = {self.conf.month} if isinstance(self.conf.month, int) else set(self.conf.month)

        # determine area folder
        if self.conf.area is None:
            area_name = "area(x=unlimited,y=unlimited)"
        else:
            a = self.conf.area
            area_name = f"area(x=[{a['xmin']}-{a['xmax']}],y=[{a['ymin']}-{a['ymax']}])"
        area_folder = self.storage.persistent_dir() / area_name
        area_folder.mkdir(exist_ok=True)

        # in the area folder, check for missing years/months
        missing_files: set[tuple[int, int, Path]] = set()
        for year, month in product(years, months):
            p = area_folder / get_filename(year, month)
            if p.exists():
                if p.is_file():
                    pass  # cache hit
                else:
                    raise ValueError(f"Expected a file: {p}")
            else:
                missing_files.add((year, month, p))

        if missing_files:
            incomplete_years = {t[0] for t in missing_files}
            print(f"Missing or incomplete years: {', '.join(map(str, incomplete_years))}")
            missing_months = {t[1] for t in missing_files}
            print(missing_months)

            # Manually create ThreadPoolExecutor with high number of workers
            executor = ThreadPoolExecutor(max_workers=128)
            loop = asyncio.get_event_loop()
            loop.set_default_executor(executor)
            loop.run_until_complete(
                run_pipeline(
                    year=incomplete_years,
                    month=missing_months,
                    folder=area_folder,
                    area=self.conf.area,
                    api_key=self.conf.api_key,
                    processing_workers=2
                )
            )

        return CerraStoreAsset(path=area_folder)
