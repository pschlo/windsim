from collections.abc import Generator
from pathlib import Path
from typing import override
import time
import logging
from dataclasses import dataclass
from dask.distributed import Client, LocalCluster
from contextlib import contextmanager

from planner import Recipe, Asset, inject, DataAsset

log = logging.getLogger(__name__)


class DaskClusterAsset(Asset):
    pass


@dataclass
class DaskClusterConfAsset(Asset):
    num_workers: int | None
    threads_per_worker: int | None
    memory_per_worker: str | None
    dashboard_address: str


class DaskClusterRecipe(Recipe[DaskClusterAsset]):
    _makes = DaskClusterAsset

    config: DaskClusterConfAsset = inject()

    @override
    @contextmanager
    def make(self):
        log.info("Creating Dask cluster")
        _dask_start = time.perf_counter()
        # Client is automatically registered as the default scheduler;
        # see https://distributed.dask.org/en/latest/client.html#dask
        # TODO: Catch CommClosedError
        with (
            LocalCluster(
                n_workers=self.config.num_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=self.config.memory_per_worker,
                dashboard_address=self.config.dashboard_address,
            ) as cluster,
            Client(cluster) as client
        ):
            _duration = round(time.perf_counter() - _dask_start, 2)
            log.info(f"Creating Dask cluster took {_duration} seconds")

            try:
                yield DaskClusterAsset()
            finally:
                log.info(f"Shutting down Dask cluster")
                cluster.close()
                client.close()
