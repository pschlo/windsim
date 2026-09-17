# windsim

Scalable Python framework for wind turbine noise prediction and mapping.

Windsim predicts noise at individual receiver locations and across spatial grids
using ISO 9613-2 based sound propagation calculations. It combines turbine sound
power spectra, elevation data, and parallel numerical computation to produce
noise maps and numerical results for analysis and custom Python workflows.
Shadow simulation is experimental and is not available through the CLI yet.

![Example wind turbine noise map](https://github.com/user-attachments/assets/447cfac7-e605-4c67-b202-8fa01ab86cc7)

*Example noise simulation output.*

## Highlights

- **Point and grid predictions:** calculate sound pressure levels at specific
  receivers or on a grid with configurable extent, spacing, and height.
- **Frequency-dependent noise modeling:** supply octave-band turbine sound power
  spectra, with an option for the German *Interimsverfahren*.
- **Automatic terrain handling:** download and reuse missing FABDEM elevation
  tiles, combine them, and reproject them to the simulation's coordinate system.
- **Wind data utilities:** separate Python workflows retrieve CERRA wind speed,
  direction, and turbulent kinetic energy at multiple heights and prepare
  compressed NetCDF datasets. CERRA is not used by the default noise workflow.
- **Maps and numerical data:** generate PNG maps, export NetCDF datasets, and
  access labeled Xarray results from Python.
- **Parallel computation:** configure Dask workers, threads, memory limits, and
  chunk sizes for larger receiver grids.

See [model assumptions and limitations](docs/noise-model.md) for the scope of the
calculations and the status of experimental features.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), download
and extract the [source archive](https://github.com/pschlo/windsim/archive/refs/heads/main.zip),
then run this command in the extracted directory:

```console
uv run windsim noise
```

This runs the bundled example and saves a noise map to
`example_repository/projects/default/noise_output/output.png`. No map-provider
account or configuration edits are needed. uv installs the dependencies and can
download Python 3.12 or newer if needed.

### Your own data

With a prepared data repository, install and run directly from GitHub:

```console
uvx https://github.com/pschlo/windsim/archive/refs/heads/main.zip noise --root path/to/data-repository --project my-project
```

Append `--help` to see command options. See the [input guide](docs/input-repository.md)
for NetCDF exports, optional geographic background tiles, and configuration.
The archive commands follow the rolling `main` branch; use an immutable source
version and retain the lockfile, inputs, and settings for reproducible work.

## Terrain and wind data

The standard noise workflow automatically prepares FABDEM elevation data for
the selected area. It reuses existing tiles and downloads missing tile
collections, then combines, reprojects, and clips the terrain data. The bundled
example already includes its elevation tile.

The project also includes a CERRA height-level wind data pipeline: retrieve
monthly GRIB files from Copernicus CDS, select a geographic area, and convert,
chunk, and compress the results as NetCDF for reuse. These utilities are available
through Python; wind-dependent acoustics and general weather inputs are not
integrated into the default noise simulation. See [data sources and processing](docs/data-sources.md)
for variables, caching, prerequisites, and current boundaries.

## Input repository

Simulation data is organized separately from the source code:

```text
data-repository/
├── shared/
│   ├── config.toml          # simulation, computation, and output settings
│   └── fabdem/              # elevation tiles, downloaded when missing
└── projects/
    └── my-project/
        ├── setup.toml       # turbine models, turbines, and specific receivers
        └── noise_output/    # generated map output
```

`--root` selects the data repository; `--project` selects a project inside it.
The defaults are `./example_repository` and `default`. The CLI reads a shared
configuration and a separate setup file for each project. NetCDF exports use
the configured file-output folder rather than `noise_output/`.

Start with the [example setup](example_repository/projects/default/setup.toml)
and [configuration](example_repository/shared/config.toml). The
[input guide](docs/input-repository.md) explains the supported fields,
longitude/latitude order, heights, elevations, and sound power spectra.

## Architecture and customization

Windsim uses [Planner](https://github.com/pschlo/planner), also developed by Peter
Schlosshan, to assemble workflows from assets and recipes. Input preparation,
terrain loading, simulation, and output generation are separate processing
stages. Custom Python workflows can supply existing assets or substitute
providers while preserving the expected data structures.

| Library | Role |
| --- | --- |
| Planner | Resolve stage dependencies, assemble execution plans, and manage resources and storage |
| Xarray | Represent labeled multidimensional inputs and results |
| Dask | Execute lazy numerical calculations in parallel |

Possible extensions include custom input adapters, terrain providers, and
exports. These require Python code and explicit provider selection. The
[customization guide](docs/extending.md) includes a minimal executable example
and explains the constraints on replacing recipes.

## Research background

The original noise implementation, model assumptions, software design, and
performance evaluation are documented in Peter Schlosshan's bachelor's thesis
at RWTH Aachen University (November 2024):

[**Efficient and Scalable Sound Propagation Modeling for Predicting Wind Turbine Noise Immission**](https://ths.rwth-aachen.de/wp-content/uploads/sites/4/thesis_Schlosshan.pdf).

The thesis compares the original implementation with WindPRO. Its tested noise
scenario ran 22–108 times faster, with barrier attenuation disabled; a separate
multicore evaluation showed approximately sixfold speedup. These are historical
results for the configurations in Sections 6.1–6.2, not benchmarks of the current
version. Agreement with another implementation is not validation against field
measurements. Use the thesis when citing the methods and original evaluation,
and identify the software version and inputs when reporting new results.

## Current scope

Noise prediction is the primary CLI workflow. Shadow and Harmonoise code are
experimental; barrier inputs are not fully integrated. The default preparation
uses fixed atmospheric and operating assumptions, and the TA Lärm aggregation
does not implement the full assessment procedure described in the thesis.
Output generation computes the selected arrays in memory, so workload size
still matters when using chunked computation.

See [model assumptions and limitations](docs/noise-model.md) before changing
acoustic options or interpreting results.

## Development

```console
uv sync --locked
uv run python -m pytest
uv build
```

Planner is installed from its rolling `main` archive, with an artifact hash in
the lockfile. Run `uv lock --refresh-package planner` to deliberately refresh
that dependency. See [customization](docs/extending.md) before changing the
default recipe bundle.
