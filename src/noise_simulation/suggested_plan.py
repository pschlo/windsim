from pathlib import Path

from planner import Planner, StaticRecipe, assets as common_assets

from . import assets as noise_assets
from .config import Config


from planner import recipes as common_recipes, assets as common_assets, RecipeBundle
from . import assets as noise_assets, recipes as noise_recipes


RECIPE_BUNDLE = RecipeBundle([
    common_recipes.ExistingTurbinesJson,
    common_recipes.ScenariosJson,
    common_recipes.TurbineTypesJson,
    common_recipes.Fabdem,
    common_recipes.DaskCluster,
    common_recipes.ReceiversJson,

    noise_recipes.Area,
    noise_recipes.TransformedTurbines,
    noise_recipes.RawTurbines,
    noise_recipes.TurbineTypes,
    noise_recipes.Frequencies,
    noise_recipes.WorkingCrs,
    noise_recipes.Chunksize,
    noise_recipes.AreaOfInterest,
    noise_recipes.FabdemConf,
    noise_recipes.Elevation,
    noise_recipes.Receivers,
    noise_recipes.FullTurbines,
    noise_recipes.TimesliceDurations,
    noise_recipes.AtmosphericCoefficient,
    noise_recipes.ExposureTimes,
    noise_recipes.Coarseness,
    noise_recipes.SoundPowerLevels,
    noise_recipes.AWeighting,
    noise_recipes.ReceiverGroups,
    noise_recipes.NoiseSimulation,
    noise_recipes.NoiseOutput
])


def run_simulation(root: Path | str, project: str, config_path: Path | str | None = None) -> noise_assets.NoiseOutput:
    """Runs a noise simulation.

    Uses the `suggested_recipes` to create a `Plan`, which is then executed.
    """
    config = Config.load(full_path=config_path).data

    plan = (
        Planner()
        .add(RECIPE_BUNDLE)
        .add(StaticRecipe(noise_assets.Config(config)))
        .add(StaticRecipe(common_assets.DaskClusterConf(
            num_workers=config.computation.num_workers,
            threads_per_worker=config.computation.threads_per_worker,
            memory_per_worker=config.computation.memory_per_worker,
            dashboard_address=config.computation.dashboard_address
        )))
        .plan(
            noise_assets.NoiseOutput,
            root=root,
            project=project
        )
    )

    with plan.run() as asset:
        return asset
