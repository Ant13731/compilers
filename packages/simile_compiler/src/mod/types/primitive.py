from __future__ import annotations
from dataclasses import dataclass

from src.mod.types.traits import Trait
from src.mod.types.base import BaseType


@dataclass(kw_only=True, frozen=True)
class NoneType_(BaseType):
    """Intended for statements without a type, not expressions. For example, a while loop node doesn't have a type."""


@dataclass(kw_only=True, frozen=True)
class StringType(BaseType):
    pass


@dataclass(kw_only=True, frozen=True)
class IntType(BaseType):
    pass


@dataclass(kw_only=True, frozen=True)
class FloatType(BaseType):
    pass
