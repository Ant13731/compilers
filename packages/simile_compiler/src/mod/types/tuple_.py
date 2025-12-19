from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable

from src.mod.types.base import BaseType, BoolType
from src.mod.types.traits import Trait
from src.mod.types.primitive import NoneType_, IntType


@dataclass(kw_only=True, frozen=True)
class TupleType(BaseType):
    items: tuple[BaseType, ...]

    def __post__init__(self):
        for item in self.items:
            if not isinstance(item, BaseType):
                raise TypeError(f"TupleType items must be BaseType instances, got {type(item)}")


@dataclass(kw_only=True, frozen=True)
class PairType(TupleType):

    def __init__(self, left: BaseType, right: BaseType) -> None:
        super().__init__(items=(left, right))

    @property
    def left(self) -> BaseType:
        return self.items[0]

    @property
    def right(self) -> BaseType:
        return self.items[1]
