# Extending Windsim

Windsim separates input preparation, simulation, and output into recipes. Custom
Python workflows can reuse these stages or supply alternatives. Start with a
custom report or output stage: it lets you reuse the acoustic model and existing
inputs while choosing what to present or export.

## How the pieces fit together

- [Planner](https://github.com/pschlo/planner) resolves dependencies between
  **assets** (inputs or intermediate results) and **recipes** (their producers).
  It executes recipes in dependency order and manages resource cleanup.
- [xarray](https://docs.xarray.dev/) holds labeled arrays and datasets, including
  dimensions such as turbine, receiver, and frequency.
- [Dask](https://docs.dask.org/) executes the lazy numerical task graphs, using
  the workers configured by Windsim. Planner's recipe execution itself is
  sequential.

A recipe declares the asset type it produces with `_makes`, marks its input
fields with `inject()`, and implements `make()`. `StaticRecipe` provides an asset
you already have. The default noise workflow is assembled in
[`suggested_plan.py`](../src/windsim/models/noise/suggested_plan.py).

## A small custom workflow

This example supplies a turbine list and builds a count from Windsim's existing
`TurbinesDictAsset`. It does not run a simulation or download data. Save it as
`count_turbines.py` and run `uv run python count_turbines.py` from the checkout
after installing its dependencies.

```python
from planner import DataAsset, Planner, Recipe, StaticRecipe, inject
from windsim.common.assets.turbines_dict import TurbinesDictAsset


class TurbineCount(DataAsset[int]):
    pass


class CountTurbines(Recipe[TurbineCount]):
    _makes = TurbineCount
    turbines: TurbinesDictAsset = inject()

    def make(self):
        return TurbineCount(len(self.turbines.d))


turbines = TurbinesDictAsset([
    {
        "name": "example",
        "model": "example-model",
        "status": "operating",
        "position_lonlat": [6.0, 50.0],
    }
])

plan = (
    Planner()
    .add(StaticRecipe(turbines))
    .add(CountTurbines)
    .plan(TurbineCount)
)
with plan.run() as count:
    print(count.d)  # 1
```

## Choosing alternative providers

Adding another recipe for the same asset does **not** automatically override the
original. Equally suitable providers cause a planning error. Build a curated
recipe bundle that includes only the desired provider, or register an
alternative with a more specific dependency-path context using
`Planner.add(..., context=...)`. Context selection is part of Planner's API;
check the chosen plan before running a modified workflow.

The CLI uses the default bundle. Custom recipe selection currently requires a
Python workflow; it is not a CLI plugin setting or a promised stable plugin API.

## Useful extension points

- **Input adapters:** produce the existing turbine or receiver dictionary assets
  from another format instead of extracting them from `setup.toml`.
- **Terrain:** provide an alternative elevation asset, preserving the required
  coordinate reference system, coordinates, units, and array layout.
- **Reports and exports:** consume `NoiseSimulationAsset` in a custom output
  recipe. Its results are lazy, so compute only the quantities you need while
  keeping required resources alive. The
  [default output recipe](../src/windsim/models/noise/output/output_asset.py)
  shows how to declare the Dask cluster and other required assets as dependencies.
- **Model inputs:** alternative recipes can supply exposure times, time-slice
  durations, or atmospheric coefficients. These require code changes to the
  workflow; they are not general-purpose configuration options today. A custom
  schedule also needs to address the current day-only sound-power selection
  and output time-slice assumptions; see [model limitations](noise-model.md).

An asset's type alone does not describe its full data contract. Preserve the
expected dictionary fields, dimensions, coordinates, units, and eager/lazy data
behavior, and inspect downstream consumers when replacing a stage. Verify a
custom workflow against the dependency versions in the checkout's lockfile.

Planner is pinned to a tested [source commit](https://github.com/pschlo/planner/tree/3105d25ade9d2f390ac1a3eeb58ed7257ab5f58b),
with its archive hash in the lockfile. To upgrade, change the commit in the
dependency URL in `pyproject.toml`, run `uv lock`, and verify the workflow.
