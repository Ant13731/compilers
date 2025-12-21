from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable

from src.mod.types.base import BaseType
from src.mod.types.traits import Trait, TraitCollection


@dataclass
class TupleType(BaseType):
    items: tuple[BaseType, ...]

    def __post__init__(self):
        for item in self.items:
            if not isinstance(item, BaseType):
                raise TypeError(f"TupleType items must be BaseType instances, got {type(item)}")

    def _is_eq_type(self, other: BaseType) -> bool:
        if not isinstance(other, TupleType):
            return False
        if len(self.items) != len(other.items):
            return False

        for self_item, other_item in zip(self.items, other.items):
            if not self_item._is_eq_type(other_item):
                return False
        return True

    def _is_subtype(self, other: BaseType) -> bool:
        if not isinstance(other, TupleType):
            return False
        if len(self.items) != len(other.items):
            return False

        for self_item, other_item in zip(self.items, other.items):
            if not self_item._is_subtype(other_item):
                return False
        return True

    def _is_sub_traits(self, other: BaseType) -> bool:
        if self.trait_collection.empty_trait is not None:
            return True
        raise NotImplementedError

    @classmethod
    def enumeration(cls, element_types: list[BaseType]) -> TupleType:
        """Create a set from an enumeration of elements of a specific type."""
        if element_types == []:
            return cls(items=())

        return cls(items=tuple(element_types))


@dataclass
class PairType(TupleType):

    def __init__(self, left: BaseType, right: BaseType, *, trait_collection: TraitCollection | None = None) -> None:
        if trait_collection is None:
            trait_collection = TraitCollection()

        super().__init__(items=(left, right), trait_collection=trait_collection)

    @property
    def left(self) -> BaseType:
        return self.items[0]

    @property
    def right(self) -> BaseType:
        return self.items[1]

    @classmethod
    def maplet(cls, key_type: BaseType, value_type: BaseType) -> PairType:
        """Create a PairType representing a maplet from key_type to value_type."""
        return cls(left=key_type, right=value_type)
