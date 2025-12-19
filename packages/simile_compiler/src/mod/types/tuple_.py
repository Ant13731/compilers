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

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, TupleType):
            return False
        if len(self.items) != len(other.items):
            return False

        for f, o in zip(self.items, other.items):
            if not f._eq_type(o, substitution_mapping):
                return False
        return True

    # TODO add in subtype check

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return TupleType(
            items=tuple(item._replace_generic_types(lst) for item in self.items),
            traits=self.traits,
        )


@dataclass(kw_only=True, frozen=True)
class PairType(TupleType):

    def __init__(self, *, left: BaseType, right: BaseType, traits: list[Trait] | None = None) -> None:
        if traits is None:
            traits = []
        super().__init__(items=(left, right), traits=traits)

    @property
    def left(self) -> BaseType:
        return self.items[0]

    @property
    def right(self) -> BaseType:
        return self.items[1]

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return PairType(
            left=self.left._replace_generic_types(lst),
            right=self.right._replace_generic_types(lst),
            traits=self.traits,
        )
