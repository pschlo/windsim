import shutil
import subprocess
from collections.abc import Collection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict
import asyncio

import numpy as np
import xarray as xr

from .cerra_client import CerraClient, AtomicRequest
from ..common.copernicus import Result


class Area(TypedDict):
    xmin: float
    xmax: float
    ymin: float
    ymax: float


async def run_pipeline(
    year: int | Collection[int],
    month: int | Collection[int] = range(1, 13),
    *,
    folder: Path | str,
    area: Area | None = None,
    temp_folder: Path | str | None = None,
    api_key: str | None = None,
    request_workers: int = 10,
    processing_workers: int = 1,
    max_waiting_results: int | None = None,  # Max submitted-but-not-downloaded
    skip_existing: bool = True
):
    """
    - download as GRIB
    - convert to NetCDF using xarray
    - Rechunk using nccopy


    - send multiple download requests so that they are queued
    - Then spawn multiple processes, each:
        - download GRIB file
        - convert to netCDF
        - rechunk and compress
    """

    # Resolve paths
    folder = Path(folder).resolve()
    folder.mkdir(exist_ok=True)
    if temp_folder is not None:
        temp_folder = Path(temp_folder).resolve()
        temp_folder.mkdir(exist_ok=True)


    # Ensure nccopy is installed
    try:
        subprocess.run(['nccopy'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("Cannot find nccopy")

    # Create semaphore to track not-yet-downloaded/waiting results
    if max_waiting_results is None:
        max_waiting_results = request_workers
    if max_waiting_results < request_workers:
        raise ValueError(f"max_waiting_results must be at least request_workers")

    # Queues for work items
    submit_queue = asyncio.Queue[AtomicRequest]()  # All atomic requests
    download_queue = asyncio.Queue[Result]()  # Ready to download
    
    # Semaphore tracks "in-flight" requests across stages
    waiting_results_sem = asyncio.Semaphore(max_waiting_results)

    client = CerraClient(api_key=api_key)

    # Fill submit queue
    for req in client.prepare_wind_request(year=year, month=month):
        await submit_queue.put(req)

    async def submit_worker():
        """Submit requests, respecting in-flight limit."""
        while True:
            req = await submit_queue.get()

            # Wait if too many in flight
            await waiting_results_sem.acquire()  # Blocks if at limit

            print(f"Submitting atomic request: {req}")

            try:
                result = await client.submit(req)
                print(f"Request ready: {req}")
                await download_queue.put(result)
            except Exception:
                print(f"Atomic request failed: {req}")
                waiting_results_sem.release()  # Release on error
                raise
            finally:
                submit_queue.task_done()

    async def download_worker(tmpdir: Path):
        """Download results and release semaphore."""
        while True:
            result = await download_queue.get()
            print(f"Processing result: {result}")

            try:
                await _download_and_process(
                    result,
                    tmpdir=tmpdir,
                    folder=folder,
                    area=area,
                    skip_existing=skip_existing
                )
                print(f"Finished processing result: {result}")
            except Exception as e:
                print(f"Processing failed for {result.name}: {e}")
            finally:
                waiting_results_sem.release()  # Always release
                download_queue.task_done()
    
    # Start workers
    with TemporaryDirectory(dir=temp_folder) as tmpdir:
        tmpdir = Path(tmpdir)

        submit_tasks = [
            asyncio.create_task(submit_worker())
            for _ in range(request_workers)
        ]
        download_tasks = [
            asyncio.create_task(download_worker(tmpdir)) 
            for _ in range(processing_workers)
        ]

        await submit_queue.join()
        for t in submit_tasks:
            t.cancel()

        await download_queue.join()
        for t in download_tasks:
            t.cancel()

    print("Pipeline finished")


async def _download_and_process(result: Result, *, tmpdir: Path, folder: Path, area: Area | None = None, skip_existing: bool = True):
    grib = tmpdir / f"{result.name}.grib"
    contiguous_clipped_netcdf = tmpdir / f"{result.name}-contiguous-clipped.nc"
    chunked_netcdf = tmpdir / f"{result.name}-chunked.nc"
    final_netcdf = folder / f"{result.name}.nc"

    if skip_existing and final_netcdf.exists():
        print(f"Skipping existing file: {final_netcdf.name}")
        return

    # Download GRIB file
    print(f"Downloading GRIB file: {result.name}")
    await result.download(grib, temp_folder=tmpdir)

    # Convert to unchunked, uncompressed netCDF
    print(f"Converting GRIB to spatially subset netCDF: {result.name}")
    await asyncio.to_thread(
        grib_to_netcdf, grib, contiguous_clipped_netcdf, temp_folder=tmpdir, area=area
    )
    await asyncio.to_thread(grib.unlink)

    # Chunk and compress netCDF
    print(f"Chunking and compressing netCDF: {result.name}")
    await asyncio.to_thread(
        chunk_compress_netcdf, contiguous_clipped_netcdf, chunked_netcdf, temp_folder=tmpdir
    )
    await asyncio.to_thread(contiguous_clipped_netcdf.unlink)

    print(f"Finished processing: {result.name}")
    await asyncio.to_thread(
        shutil.move, chunked_netcdf, final_netcdf
    )


def grib_to_netcdf(source: Path | str, dest: Path | str, temp_folder: Path | str | None = None, area: Area | None = None):
    """Convert GRIB file to unchunked, uncompressed netCDF file."""
    # Resolve paths
    source = Path(source).resolve()
    dest = Path(dest).resolve()
    if temp_folder is not None:
        temp_folder = Path(temp_folder).resolve()
        temp_folder.mkdir(exist_ok=True)

    # Open GRIB file.
    # Do not chunk spatial dims as full grid needs to be loaded anyway.
    ds = xr.open_dataset(
        source,
        engine="cfgrib",
        decode_timedelta=True,
        chunks=dict(time=1, x=-1, y=-1),
        backend_kwargs=dict(
            indexpath='',  # disable creation of index file
        )
    )

    # Clip to area
    if area is not None:
        ds = clip_dataset(ds, area=area, buffer=3)

    with TemporaryDirectory(dir=temp_folder) as tmpdir:
        tmpdir = Path(tmpdir)
        tmpdest = tmpdir / dest.name

        # Convert to netCDF without specifying encoding, thus preserving all metadata.
        ds.to_netcdf(tmpdest, engine="netcdf4", format="NETCDF4")
        shutil.move(tmpdest, dest)


def nccopy(
    source: Path | str,
    dest: Path | str,
    compression_level: int | None = None,
    chunking: dict[str, int] | None = None,
    copy_buffer: int | str | None = None,
    chunk_cache: int | str | None = None,
    cache_elems: int | str | None = None
):
    source = Path(source).resolve()
    dest = Path(dest).resolve()

    # Build full chunking specs
    print("Reading source file metadata")
    full_chunking: dict[str, int | None] | None = None
    if chunking is not None:
        full_chunking = {}
        with xr.open_dataset(
            source,
            engine="netcdf4",
            decode_times=False,
            decode_timedelta=False,
        ) as ds:
            for dim in ds.dims:
                assert isinstance(dim, str)
                full_chunking[dim] = chunking.get(dim, None)

    # Build command
    cmd = ['nccopy']

    cmd += ['-k', 'netCDF-4']

    if copy_buffer is not None:
        cmd += ['-m', str(copy_buffer)]
    if chunk_cache is not None:
        cmd += ['-h', str(chunk_cache)]
    if cache_elems is not None:
        cmd += ['-e', str(cache_elems)]

    if compression_level is not None:
        cmd += ['-d', str(compression_level)]

    if full_chunking is not None:
        print("Chunking:", full_chunking)
        chunk_str = ','.join(
            f"{dim}/{size if size is not None else ''}"
            for dim, size in full_chunking.items()
        )
        cmd += ['-c', chunk_str]

    cmd += [str(source), str(dest)]

    print(f"Running nccopy command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def chunk_compress_netcdf(source: Path | str, dest: Path | str, temp_folder: Path | str | None = None):
    """Chunk and compress netCDF file."""
    # Resolve paths
    source = Path(source).resolve()
    dest = Path(dest).resolve()
    if temp_folder is not None:
        temp_folder = Path(temp_folder).resolve()
        temp_folder.mkdir(exist_ok=True)

    with TemporaryDirectory(dir=temp_folder) as tmpdir:
        tmpdir = Path(tmpdir)
        tmpdest = tmpdir / dest.name

        nccopy(
            source,
            tmpdest,
            compression_level=4,
            chunking=dict(time=24, x=32, y=32),
            copy_buffer='1G',
            chunk_cache=0,
            cache_elems=0
        )
        shutil.move(tmpdest, dest)


def clip_dataset(ds: xr.Dataset, area: Area, buffer: int = 0):
    """Clip the dataset so that it is contained in the given area.
    
    Optionally add some buffer to the area.
    """
    xmin, ymin, xmax, ymax = area['xmin'], area['ymin'], area['xmax'], area['ymax']
    lat, lon = ds['latitude'], ds['longitude']

    # Create mask for the area of interest
    mask = (lat >= ymin) & (lat <= ymax) & (lon >= xmin) & (lon <= xmax)

    # Find the bounding indices
    y_inds, x_inds = np.where(mask)

    if y_inds.size == 0 or x_inds.size == 0:
        raise ValueError("No points found within the specified lat/lon bounds.")

    # Find min and max index bounds
    xmin_idx = max(0, x_inds.min() - buffer)
    ymin_idx = max(0, y_inds.min() - buffer)
    xmax_idx = x_inds.max() + buffer
    ymax_idx = y_inds.max() + buffer

    # Subset the DataArray using these index bounds
    ds = ds.isel(y=slice(ymin_idx, ymax_idx + 1), x=slice(xmin_idx, xmax_idx + 1))

    return ds
