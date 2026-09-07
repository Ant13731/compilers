from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.mod.data.ast_ import ASTNode

if TYPE_CHECKING:
    from src.mod.data.types import BaseType

# These must be hash-safe
SimileLiteralAsPythonOrderable = int | float | str | tuple
SimileLiteralAsPython = bool | SimileLiteralAsPythonOrderable | set | None


@dataclass(frozen=True)
class BaseTrait:
    pass


@dataclass(frozen=True)
class ImmutableTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class LiteralTrait(BaseTrait):
    value: SimileLiteralAsPython


@dataclass(frozen=True)
class UndefinedTrait(BaseTrait):
    pass


@dataclass(frozen=True)
class GenericBoundTrait(BaseTrait):
    bound_type: BaseType  # can have multiple of the same trait here - multiple generic bounds mean a type union
