import logging
from .setup_logging import setup_logging
setup_logging()
from pathlib import Path

log = logging.getLogger(__name__)


def main():
    log.info("Starting")
    from windsim.models.noise.suggested_plan import run_simulation

    run_simulation(
        root=Path.cwd(),
        project="foo"
    )
