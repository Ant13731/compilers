from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, TypeVar

from src.mod.types.traits import Trait

T = TypeVar("T", bound="BaseType")


# Primitive types
@dataclass(kw_only=True, frozen=True)
class BaseType:
    """Base type for all Simile types."""

    traits: list[Trait] = field(default_factory=list)

    # Actual type methods
    def cast(self, caster: Callable[[BaseType], T]) -> T:
        """Cast the type to a different type."""
        raise NotImplementedError

    def equals(self, other: BaseType) -> BoolType:
        """Check if this type is equal to another type."""
        raise NotImplementedError

    # Helper methods
    def eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType] | None = None) -> bool:
        if substitution_mapping is None:
            substitution_mapping = {}
        return self._eq_type(other, substitution_mapping)

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        raise NotImplementedError

    def is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType] | None = None) -> bool:
        """Check if self is a sub-type of other."""
        if substitution_mapping is None:
            substitution_mapping = {}
        return self._is_sub_type(other, substitution_mapping)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        raise NotImplementedError

    def replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        """Structurally replace generic types in self according to the provided list of types."""
        new_lst = deepcopy(lst)
        return self._replace_generic_types(new_lst)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        raise NotImplementedError


# BoolType needs to be here to avoid circular imports
@dataclass(kw_only=True, frozen=True)
class BoolType(BaseType):
    pass
