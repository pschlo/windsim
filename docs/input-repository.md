# Input repository

The CLI reads a shared `config.toml` and a project-specific `setup.toml`:

```text
data-repository/
├── shared/
│   ├── config.toml
│   └── fabdem/
└── projects/
    └── my-project/
        ├── setup.toml
        └── noise_output/
```

Run a project from the source checkout with:

```console
uv run windsim noise --root path/to/data-repository --project my-project
```

`--root` defaults to `example_repository` under the current working directory;
`--project` defaults to `default`. The current CLI always reads
`shared/config.toml`; it does not merge project-specific configuration overrides.
The `fabdem/` folder holds reusable elevation tiles, and `noise_output/` holds
generated maps. Start by copying the [bundled setup](../example_repository/projects/default/setup.toml)
and [configuration](../example_repository/shared/config.toml), then edit them.

## Project setup

`setup.toml` defines turbine models, turbines, and specific receivers. For example:

```toml
[turbine_model.example]
# Frequencies: 63, 125, 250, 500, 1000, 2000, 4000, 8000 Hz.
# Illustrative sound power spectrum, not a manufacturer's specification.
sound_power_db = [89.5, 93.8, 97.2, 98.6, 99.4, 97.0, 92.6, 87.5]

[[turbine]]
name = 'T01'
model = 'example'
hub_height_m = 90
position_lonlat = [8.585, 56.295]

[[receiver]]
name = 'R01'
position_lonlat = [8.600, 56.300]
height_m = 5
```

Coordinates in `position_lonlat` are WGS84 **[longitude, latitude]**, in degrees.
Heights are in meters above the local ground. `elevation_m` is ground elevation
in meters above mean sea level; when omitted, it is interpolated from FABDEM.
The calculation adds height to ground elevation to obtain the source or receiver
altitude. Use compatible elevation references when supplying your own values.

| Section | Field | Meaning and default |
| --- | --- | --- |
| `turbine_model.<id>` | `sound_power_db` | Required: eight unweighted octave-band source sound power levels in dB re 1 pW, ordered as above |
| `turbine_model.<id>` | `manufacturer` | Optional descriptive text |
| `[[turbine]]` | `model` | Required: an existing turbine model ID |
| `[[turbine]]` | `hub_height_m` | Required: hub height above ground |
| `[[turbine]]` | `position_lonlat` | Required: longitude and latitude |
| `[[turbine]]` | `name` | Optional; defaults to `<unnamed>` |
| `[[turbine]]` | `status` | Optional metadata; defaults to `new`; does not control operating schedules or exclude turbines |
| `[[turbine]]` | `elevation_m` | Optional ground elevation; otherwise uses FABDEM |
| `[[receiver]]` | `position_lonlat` | Required: longitude and latitude |
| `[[receiver]]` | `name` | Optional; defaults to an empty string |
| `[[receiver]]` | `height_m` | Optional; otherwise uses `input.normal.default_height` |
| `[[receiver]]` | `elevation_m` | Optional ground elevation; otherwise uses FABDEM |

Supply unweighted source sound power spectra, not the desired sound pressure
level at a receiver. The model applies frequency-dependent attenuation and
A-weighting; supplying already A-weighted spectra would apply the weighting twice.
Model IDs must match turbine references; each spectrum must have eight values.
The input preparation rejects unknown fields within turbine model, turbine,
and receiver records. The standard setup
loader expects all three sections, including a receiver list even when using
only grid receivers.

## Shared configuration

The [example configuration](../example_repository/shared/config.toml) documents
the supported settings. Copy the complete file: several settings are required
and a partial file is not automatically combined with defaults.

| Section | Common settings |
| --- | --- |
| `[input]` | Enable `normal` specific receivers and/or `grid` receivers; select the area |
| `[input.normal]` | Default receiver height above ground |
| `[input.grid]` | Approximate grid spacing, height, and optional constant ground elevation |
| `[computation]` | Workers, threads, worker memory, chunk size, working CRS, and acoustic options |
| `[output]` | Enable `map`, `file`, or both |
| `[output.file]` | NetCDF destination folder |
| `[output.map]` | Map buffer, contours, labels, and tile-provider settings |
| `[debug]` | Enabled debug transformations; use `enabled = []` to disable them |

A numeric `input.area` expands the turbine bounding box by that distance on each
side, in meters. A pair specifies separate x/y buffers. Center/extent and
corner specifications are also supported; see the comments in the example.
The grid includes the area boundaries, so its actual spacing may differ slightly
from the requested spacing.

`coord_reference_system = 'local_best'` chooses a local projection for
calculations. Automatic UTM selection and explicit CRS specifications are also
supported. Computation coordinates must use approximately meter-sized units;
project setup coordinates remain WGS84 longitude/latitude.

`chunk_size` accepts `'auto'`, `'disabled'`, a positive integer, or a cap such as
`{max = 1000000}`. Worker, thread, and memory settings are optional and otherwise
determined by Dask. Numerical computation uses a local Dask cluster in the
standard workflow. See [the model guide](noise-model.md) before enabling
experimental acoustic options.

## Outputs and optional map backgrounds

For numerical exports, change the existing `[output]`
section's `assets` setting to `['file']`. Set `output.file.folder` to your
desired destination. Relative file-output paths resolve from the **current
working directory**, not the data repository. Its parent directory must exist.

The exporter writes timestamped `specific-receivers_export_*.nc` and
`grid-receivers_export_*.nc` files for the enabled receiver groups. Each contains
the selected main result (`L_AT_LT`, or `L_r` when TA Lärm is enabled).
The specific-receiver dataset also includes receiver positions and names.

Map output writes `projects/<project>/noise_output/output.png`, replacing the
previous map for that project. Without an `[output.map.tiles]` table, it uses
the computation CRS and renders without external background tiles or credentials.
The bundled example uses this mode, so `uv run windsim noise` produces a map
without configuration edits.

To add geographic background tiles, uncomment the example's
`[output.map.tiles]` table and supply your own Stadia Maps key in `api_key`.
Keep credentials out of shared source files and version control. Background
tiles require network access. Missing terrain tiles are downloaded as
FABDEM tile collections and stored under `shared/fabdem/`; downloads may be
substantially larger than the simulated area. See [data sources](data-sources.md)
for terrain preparation and the separate CERRA wind data pipeline.

From a source checkout, start Python with `uv run python` and inspect a NetCDF
result with:

```python
import xarray as xr

with xr.open_dataset('output/specific-receivers_export_<timestamp>.nc') as result:
    print(result)
    print(result['L_AT_LT'])
```

Replace the filename with an actual export. With the default `ta_laerm = false`,
`L_AT_LT` contains predicted A-weighted sound pressure levels in dB(A).
See [model assumptions](noise-model.md) for interpreting them.
