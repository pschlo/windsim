from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict, cast

from noise_simulation.coordinate_reference_systems import parse_name as parse_crs_name
from .. import chunksize_specs, crs_specs
from ..exceptions import ConfigError
from ._abstract import Section


class ComputationSection(Section):
    num_workers: int | None
    memory_per_worker: str | None
    threads_per_worker: int | None
    dashboard_address: str
    crs: crs_specs.CrsSpecification

    chunksize: chunksize_specs.ChunksizeSpecification

    interim: bool
    ta_laerm: bool
    alternative_method: bool
    consider_barriers: bool

    def __init__(self, raw: dict[str, Any]) -> None:
        super().__init__(raw)

        if 'num_workers' not in raw:
            self.num_workers = None
        else:
            self.num_workers = raw['num_workers']

        if 'memory_per_worker' not in raw:
            self.memory_per_worker = None
        else:
            self.memory_per_worker = raw['memory_per_worker']

        if 'threads_per_worker' not in raw:
            self.threads_per_worker = None
        else:
            self.threads_per_worker = raw['threads_per_worker']

        if 'dashboard_address' not in raw:
            self.dashboard_address = ":8787"
        else:
            self.dashboard_address = raw['dashboard_address']

        # We cannot yet determine the exact CRS, because knowledge of the input area is required.
        self.crs = parse_crs(raw['coord_reference_system'])

        self.chunksize = parse_chunksize(raw['chunk_size'])

        self.interim = raw['interim']
        self.ta_laerm = raw['ta_laerm']
        self.alternative_method = raw['alternative_method']
        self.consider_barriers = raw['consider_barriers']


def parse_chunksize(raw) -> chunksize_specs.ChunksizeSpecification:
    match raw:
        case int() | float():
            return chunksize_specs.Custom(int(raw))
        case 'auto':
            return chunksize_specs.Auto()
        case 'disabled':
            return chunksize_specs.Disabled()
        case _:
            if isinstance(raw, dict) and 'max' in raw:
                return chunksize_specs.CappedAuto(raw['max'])
            raise ConfigError('Invalid chunk size specification')


def parse_crs(raw) -> crs_specs.CrsSpecification:
    match raw:
        case 'UTM ETRS':
            return crs_specs.UTM_ETRS()
        case 'UTM WGS':
            return crs_specs.UTM_WGS()
        case 'local_best':
            return crs_specs.LocalBest()
        case _:
            return crs_specs.Custom(parse_crs_name(raw))
