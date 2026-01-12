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


@click.group(context_settings={
    "help_option_names": ["-h", "--help"],
    "max_content_width": 120
})
def cli():
    """windsim command line interface."""


@cli.command()
@click.option(
    "-r", "--root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=lambda: Path.cwd() / "repository",
    help="Root directory of the data repository.  [default: ./repository]",
)
@click.option(
    "-p", "--project",
    type=str,
    default="default",
    help="Project name.  [default: default]",
)
@click.option(
    "-c", "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False, file_okay=True),
    default=lambda: Path.cwd() / "config.toml",
    help="Path to the config TOML file.  [default: ./config.toml]",
)
def noise(root: Path, project: str, config_path: Path):
    """Run a noise simulation."""
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
    """Run a shadow simulation (not implemented yet)."""
    raise click.ClickException("Shadow simulation is not implemented yet.")
