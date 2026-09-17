# Terrain and wind data

Windsim contains geographic terrain preparation and a separate wind-data
ingestion pipeline. Their current integration differs:

| Source | Data | Current use |
| --- | --- | --- |
| FABDEM | Elevation rasters | Automatically prepared by the standard noise workflow |
| CERRA height levels | Wind speed, wind direction, turbulent kinetic energy | Python utilities and recipes; outside the default CLI workflow |

The standard noise workflow uses supplied turbine sound power spectra and fixed
atmospheric assumptions. It does not fetch temperature or humidity, or derive
noise predictions from CERRA wind conditions. See [model scope](noise-model.md).

## FABDEM elevation

The [terrain downloader](../src/windsim/common/data_sources/fabdem/download.py)
uses FABDEM v1-2 from the
[University of Bristol dataset](https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn).
The source rasters have one-arcsecond spacing and are organized into one-degree
tiles distributed in ZIP collections.

For the requested area, Windsim:

1. Transforms bounds to longitude/latitude and adds a small preparation buffer.
2. Finds the required tiles and checks the local cache first.
3. Downloads missing tile collections and extracts the required rasters.
4. Mosaics the rasters, reprojects them to the working CRS using bilinear
   resampling, and clips/interpolates to the requested bounds.

The [FABDEM recipe](../src/windsim/common/assets/fabdem.py) stores extracted tiles
under `shared/fabdem/` in the data repository, allowing projects to reuse them.
If every required tile is present, terrain preparation makes no download request.
ZIP downloads may be much larger than the simulation area; archives are temporary,
while the extracted rasters remain cached. Missing tiles or NaN elevation values
raise errors rather than being filled automatically.

Elevation supports source and receiver positioning. This does not mean that
the acoustic calculation follows the intervening terrain profile or includes
terrain screening. See [model limitations](noise-model.md).

## CERRA height-level wind data

The [request builder](../src/windsim/common/data_sources/cerra_heights/cerra_client.py)
targets the CDS `reanalysis-cerra-height-levels` collection. Its wind request
selects wind speed, wind direction, and turbulent kinetic energy at:

**15, 30, 50, 75, 100, 150, 200, 250, 300, 400, and 500 m.**

It requests the forecast product within the reanalysis collection, using
initialization hours 00, 03, 06, 09, 12, 15, 18, and 21 with one-, two-, and
three-hour lead times. The
[store loader](../src/windsim/common/assets/cerra_store.py) combines initialization
and lead-time axes into an hourly `time` axis using the recorded `valid_time`.
It does not interpolate these height levels to turbine hub heights.

The [Python pipeline](../src/windsim/common/data_sources/cerra_heights/pipeline.py)
queues monthly requests, downloads GRIB files, converts them with cfgrib/xarray
to NetCDF, and runs `nccopy` to produce compressed, chunked NetCDF files.
Compression uses level 4; explicit chunk sizes are `time=24`, `x=32`, and `y=32`.

Optional regional selection happens **after download**. Its `xmin`/`xmax` values
are longitude and `ymin`/`ymax` latitude in degrees, despite the x/y names. The
processing step keeps a bounding grid rectangle with a three-cell buffer;
specifying a small region does not reduce the original GRIB request.

## Storage, prerequisites, and current boundaries

The CERRA store recipe keeps monthly `year=YYYY,month=M.nc` files beneath
`shared/cerra-store/area(...)/`. An existing monthly file counts as cached;
the loader can select years/months and opens the data through xarray. This cache
does not validate source revisions or file integrity.

Retrieval requires a CDS API key supplied explicitly or through `CDS_KEY`, plus
access to the requested dataset. The conversion path requires cfgrib with its
GRIB decoder, NetCDF4 support, and the external `nccopy` executable. The latter
must be available on `PATH`; it is not installed by the Python package.

The default noise recipe bundle includes FABDEM, but does not include CERRA.
Using CERRA requires a custom Python workflow; there is no CLI wind-data option
or automatic connection to turbine sound power. See [customization](extending.md).

A complete store cache avoids retrieval. Direct pipeline runs can submit
requests before checking existing output files. Processing errors currently get
printed, so verify that all requested monthly outputs exist and can be opened;
the pipeline's completion message alone does not establish that they succeeded.
