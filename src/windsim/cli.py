import logging
from .setup_logging import setup_logging
setup_logging(
    logging.INFO,
    loggers=[
        "planner",
    ]
)
from pathlib import Path
import click


log = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """windsim command line interface."""


@cli.command()
@click.option(
    "--root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=lambda: Path.cwd() / "resources" / "repository",
    show_default=True,
    help="Root directory of the repository.",
)
@click.option(
    "--project",
    type=str,
    default="foo",
    show_default=True,
    help="Project name.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    default=lambda: Path.cwd() / "noise_config.toml",
    show_default=True,
    help="Path to the noise config TOML file.",
)
def noise(root: Path, project: str, config_path: Path):
    """Run the noise simulation."""
    log.info("Starting noise simulation")
    from windsim.models.noise.suggested_plan import run_simulation

    run_simulation(
        root=root,
        project=project,
        config_path=config_path,
    )


# Placeholder for later:
@cli.command()
def shadow():
    """Run the shadow simulation (not implemented yet)."""
    raise click.ClickException("Shadow simulation is not implemented yet.")
