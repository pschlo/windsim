# Noise model and current scope

The standard [noise workflow](../src/windsim/models/noise/suggested_plan.py)
prepares turbine and receiver data, constructs an Xarray/Dask calculation graph,
and computes selected results for output. Turbine octave-band sound power
spectra are inputs; the model predicts sound pressure at receivers.

## Calculations

The [simulation recipe](../src/windsim/models/noise/model.py) uses ISO 9613-2 based
calculations for geometrical divergence, atmospheric absorption, and ground
effects, then sums contributions energetically and applies A-weighting.
The frequency bands are 63, 125, 250, 500, 1000, 2000, 4000, and 8000 Hz.

`computation.interim = true` selects the German *Interimsverfahren* adjustments,
including fixed ground attenuation and directivity correction. The default
example enables this option. Setting `alternative_method = true` selects the
alternative ground method and cannot be combined with `interim = true`.

The default output is `L_AT_LT`, the implementation's long-term A-weighted level.
Its meteorological correction is currently fixed to zero. If `ta_laerm = true`,
the exporter selects `L_r` from the time/exposure aggregation implementation.
The latter option requires additional assessment-method work as described below.

## Default assumptions

- Turbines are represented as sources at hub height, using user-supplied spectra.
- Atmospheric absorption coefficients are selected for 10 °C and 70% relative
  humidity. These are currently fixed in the preparation recipe.
- Sound power preparation currently selects only the `day` slice. Time durations
  are hardcoded to 16/8 hours, and exposure preparation assumes continuous
  operation; these do not constitute configurable operating schedules.
- Source directivity, meteorological correction, and miscellaneous attenuation
  are fixed to zero in the current calculation functions.
- With the regular ground method, the preparation recipe supplies constant
  ground factors of 0.8 rather than a land-cover-derived ground model.
- Elevation data determines source and receiver altitudes. The current projected
  distance and average-height calculations do not follow the intervening terrain
  profile, and terrain screening is not integrated into the standard workflow.

These assumptions are implemented in [input preparation](../src/windsim/models/noise/input)
and the [ISO calculation functions](../src/windsim/models/noise/computation/iso9613).
Some can be changed in custom Python workflows; they are not all configuration
options. See [customization](extending.md).

## Experimental and incomplete features

| Feature | Current boundary |
| --- | --- |
| Shadow simulation | Internal code exists, but the CLI command raises an unimplemented error and the internal workflow needs integration work |
| Barriers | Keep `consider_barriers = false` in the standard workflow; the enabled path references a barrier asset that the recipe does not declare or receive |
| Harmonoise | Exploratory implementation; the standard simulation recipe does not select it |
| TA Lärm | Energy/time aggregation exists, but the tonal, impulsive, and increased-sensitivity corrections described in thesis Section 4.5 are absent; the default time preparation also needs completion |

The standard workflow creates a local Dask cluster. It does not expose a remote
cluster connection through the CLI. Output generation computes selected arrays
in memory before exporting or plotting; chunking does not eliminate output
memory requirements.

## Research and evaluation

Peter Schlosshan's [2024 bachelor's thesis](https://ths.rwth-aachen.de/wp-content/uploads/sites/4/thesis_Schlosshan.pdf)
explains the original models, implementation, and evaluation. Section 4 describes
the acoustic methods; Section 5 describes vectorization, Xarray, and Dask;
Section 6 compares the original implementation with WindPRO; Section 7.2
discusses limitations and future work.

The benchmark scenario uses four turbines and multiple receiver-grid resolutions,
with barrier attenuation disabled. Sections 6.1–6.2 report 22–108 times faster
execution than WindPRO for the tested noise scenario and approximately sixfold
speedup in a separate multicore evaluation. Runtime and agreement results apply
to that setup and the original implementation. They are not a current-version
benchmark, field validation, or evidence that every experimental feature is ready
to use. For new studies, record the source version, input spectra, configuration,
and active assumptions alongside results.
