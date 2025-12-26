from __future__ import annotations

from dataclasses import dataclass


class ChunksizeSpecification:
    pass


@dataclass
class Custom(ChunksizeSpecification):
    value: int

@dataclass
class Auto(ChunksizeSpecification):
    pass

@dataclass
class CappedAuto(ChunksizeSpecification):
    max: int

@dataclass
class Disabled(ChunksizeSpecification):
    pass
