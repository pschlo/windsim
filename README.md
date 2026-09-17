# windsim

**Scalable Python framework for wind turbine noise prediction and mapping.**

Predict sound pressure levels at specific receivers or across spatial grids using
ISO 9613-2 based acoustics, turbine sound power spectra, and elevation data.

[Quick start](#quick-start) · [Input guide](docs/input-repository.md) · [Thesis](https://ths.rwth-aachen.de/wp-content/uploads/sites/4/thesis_Schlosshan.pdf)

<p align="center">
  <a href="https://github.com/user-attachments/assets/447cfac7-e605-4c67-b202-8fa01ab86cc7">
    <img width="600" src="https://github.com/user-attachments/assets/447cfac7-e605-4c67-b202-8fa01ab86cc7" alt="Example noise map showing turbines, receivers, and predicted sound pressure contours" />
  </a>
  <br />
  <sub>Example noise map: predicted A-weighted sound pressure levels in dB(A).
    <a href="https://github.com/user-attachments/assets/447cfac7-e605-4c67-b202-8fa01ab86cc7">View full size</a>
  </sub>
</p>

## Highlights

- **Point and grid predictions** with configurable extent, spacing, and height.
- **Octave-band acoustics**, including the German *Interimsverfahren* option.
- **Automatic FABDEM terrain preparation:** download, cache, combine, and reproject
  elevation tiles. The example's tile is included.
- **CERRA wind data utilities:** retrieve wind speed, direction, and turbulent
  kinetic energy and prepare NetCDF datasets through Python, separately from noise
  simulation. See [data sources](docs/data-sources.md).
- **PNG maps, NetCDF exports, and labeled Xarray results.**
- **Parallel Dask computation** with configurable workers and chunk sizes.

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
These archive commands follow `main`; pin a source version for repeatable work.

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
The defaults are `./example_repository` and `default`.

Start with the [example setup](example_repository/projects/default/setup.toml)
and [configuration](example_repository/shared/config.toml). The
[input guide](docs/input-repository.md) explains fields, units, and output paths.

## Architecture and customization

Windsim uses [Planner](https://github.com/pschlo/planner), also developed by Peter
Schlosshan, to connect assets and recipes for input preparation, terrain,
simulation, and output. Custom Python workflows can replace providers or add
input adapters and exports. The [customization guide](docs/extending.md) includes
an example and explains constraints on replacing recipes.

## Research background

The project originated in Peter Schlosshan's bachelor's thesis at RWTH Aachen
University (November 2024):

[**Efficient and Scalable Sound Propagation Modeling for Predicting Wind Turbine Noise Immission**](https://ths.rwth-aachen.de/wp-content/uploads/sites/4/thesis_Schlosshan.pdf).

It documents the methods, software design, and performance comparisons with
WindPRO. The [evaluation notes](docs/noise-model.md#research-and-evaluation)
explain the scope of these historical results.

## Current scope

Noise prediction is the primary CLI workflow. Shadow and Harmonoise code are
experimental; barrier inputs are incomplete. The default workflow uses fixed
atmospheric and operating assumptions. See [model assumptions and limitations](docs/noise-model.md)
before interpreting results.

## Development

```console
uv sync --locked
uv run python -m pytest
uv build
```

See [customization](docs/extending.md) for dependency and workflow details.
