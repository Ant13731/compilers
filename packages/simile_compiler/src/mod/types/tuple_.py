from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable

from src.mod.types.base import BaseType, BoolType, AnyType_
from src.mod.types.traits import Trait
from src.mod.types.primitive import NoneType_, IntType


@dataclass(kw_only=True)
class TupleType(BaseType):
    items: tuple[BaseType, ...]

    def __post__init__(self):
        for item in self.items:
            if not isinstance(item, BaseType):
                raise TypeError(f"TupleType items must be BaseType instances, got {type(item)}")

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, TupleType):
            return False
        if len(self.items) != len(other.items):
            return False

        for self_item, other_item in zip(self.items, other.items):
            if not self_item._is_eq_type(other_item, substitution_mapping):
                return False
        return True

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, TupleType):
            return False
        if len(self.items) != len(other.items):
            return False

        for self_item, other_item in zip(self.items, other.items):
            if not self_item._is_sub_type(other_item, substitution_mapping):
                return False
        return True

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return TupleType(
            items=tuple(item._replace_generic_types(lst) for item in self.items),
            traits=self.traits,
        )

    @classmethod
    def enumeration(cls, element_types: list[BaseType]) -> TupleType:
        """Create a set from an enumeration of elements of a specific type."""
        if element_types == []:
            return cls(items=())

        return cls(items=tuple(element_types))


@dataclass(kw_only=True)
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
