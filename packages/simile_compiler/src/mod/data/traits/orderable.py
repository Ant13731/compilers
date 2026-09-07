from dataclasses import dataclass

from src.mod.data.ast_ import ASTNode
from src.mod.data.traits.base import BaseTrait, SimileLiteralAsPythonOrderable


@dataclass(frozen=True)
class OrderableTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class MinTrait(BaseTrait):
    """The optimizer will need to hijack the host language's ordering systems for min/max/range comparisons.
    We only expect simile literals to be given here, so the translation between python<->simile should be one-to-one.

    Non-arithmetic types are orderable lexicographically"""

    value: SimileLiteralAsPythonOrderable


@dataclass(frozen=True)
class MaxTrait(BaseTrait):
    value: SimileLiteralAsPythonOrderable
