from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, TypeVar, TYPE_CHECKING

from src.mod.types.traits import Trait

if TYPE_CHECKING:
    from src.mod.ast_.ast_node_base import ASTNode


class SimileTypeError(Exception):
    """Custom exception for Simile type errors."""

    def __init__(self, message: str, node: ASTNode | None = None) -> None:
        message = f"SimileTypeError: {message}"
        if node is not None:
            message = f"Error {node.get_location()} (at node {node}): {message}"

        super().__init__(message)
        self.node = node


T = TypeVar("T", bound="BaseType")


# Primitive types
@dataclass(kw_only=True, frozen=True)
class BaseType:
    """Base type for all Simile types."""

    # TODO should traits be a set? we really shouldn't care about order or duplicates...
    traits: list[Trait] = field(default_factory=list)

    # Actual type methods
    def cast(self, caster: Callable[[BaseType], T]) -> T:
        """Cast the type to a different type."""
        raise NotImplementedError

    def equals(self, other: BaseType) -> BoolType:
        """Check if this type is equal to another type."""
        raise NotImplementedError

    # Helper methods
    # TODO GO THROUGH EACH CHILD METHOD AND CHECK THE TYPECHECKING FUNCTION AGAINST THE FORMAL TYPE SYSTEM IN THE SPEC
    def eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType] | None = None, check_traits: bool = False) -> bool:
        if substitution_mapping is None:
            substitution_mapping = {}
        if check_traits:
            return self._eq_type(other, substitution_mapping) and self._eq_traits(other)
        else:
            return self._eq_type(other, substitution_mapping)

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        raise NotImplementedError

    def _eq_traits(self, other: BaseType) -> bool:
        """Check whether the type would be equal when considering traits."""
        raise NotImplementedError

    def is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType] | None = None, check_traits: bool = False) -> bool:
        """Check if self is a sub-type of other (in formal type theory, whether self <= other)."""
        if substitution_mapping is None:
            substitution_mapping = {}
        if check_traits:
            return self._is_sub_type(other, substitution_mapping) and self._is_sub_traits(other)
        else:
            return self._is_sub_type(other, substitution_mapping)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        raise NotImplementedError

    def _is_sub_traits(self, other: BaseType) -> bool:
        """Check whether the type is a sub-type when considering traits."""
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
    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, BoolType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, BoolType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self
