from dataclasses import dataclass

from src.mod.data.ast_ import ASTNode
from src.mod.data.traits.base import BaseTrait, SimileLiteralAsPython


@dataclass(frozen=True)
class OneToManyTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class ManyToOneTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class TotalOnDomainTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class TotalOnRangeTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class RelationalDomainTrait(BaseTrait):
    values: set[SimileLiteralAsPython]


@dataclass(frozen=True)
class RelationalRangeTrait(BaseTrait):
    values: set[SimileLiteralAsPython]
