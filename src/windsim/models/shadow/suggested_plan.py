from pathlib import Path
from planner import RecipeBundle, StaticRecipe
from planner import Planner, StaticRecipe

from windsim.common import assets as common_assets, recipes as common_recipes
from windsim.models.noise import assets as noise_assets, recipes as noise_recipes
from windsim.models.noise.config import Config

from . import assets as shadow_assets, recipes as shadow_recipes


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
    noise_recipes.ReceiverGroups,

    shadow_recipes.FastShadow,
    shadow_recipes.Sunpos,
    StaticRecipe(shadow_assets.SunposConf()),
    shadow_recipes.AdaptedTurbines,
    shadow_recipes.ShadowResult,
])



def run_simulation(root: Path | str, project: str, config_path: Path | str | None = None) -> shadow_assets.ShadowResult:
    """Runs a shadowcast simulation.

    Uses the `suggested_recipes` to create a `Plan`, which is then executed.
    """
    config = Config.load(full_path=config_path).data

    plan = (
        Planner()
        .add(StaticRecipe(noise_assets.Config(config)))
        .add(StaticRecipe(common_assets.DaskClusterConf(
            num_workers=config.computation.num_workers,
            threads_per_worker=config.computation.threads_per_worker,
            memory_per_worker=config.computation.memory_per_worker,
            dashboard_address=config.computation.dashboard_address
        )))
        .add(RECIPE_BUNDLE)
        .plan(shadow_assets.ShadowResult, root=root, project=project)
    )

    with plan.run() as asset:
        return asset
