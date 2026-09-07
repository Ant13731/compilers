from dataclasses import dataclass

from src.mod.data.ast_ import ASTNode
from src.mod.data.traits.base import BaseTrait, SimileLiteralAsPython


@dataclass(frozen=True)
class DomainTrait(BaseTrait):
    values: set[SimileLiteralAsPython]


@dataclass(frozen=True)
class IterableTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class UniqueTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class EmptyTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class SizeTrait(BaseTrait):
    size: int


@dataclass(frozen=True)
class TotalTrait(BaseTrait):
    pass
