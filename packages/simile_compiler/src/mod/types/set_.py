from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable

from src.mod.types.traits import Trait
from src.mod.types.base import BaseType, BoolType
from src.mod.types.primitive import NoneType_, IntType
from src.mod.types.tuple_ import PairType


# TODO we basically need a SetSimulator that will return the expected type, element type, and traits when executing a set operation
# Then we need a code generator that will follow through on the simulator's typed promise - maybe make a mirror class that outputs generated code instead of types?
# Whats the cleanest way to do this?
#
# At codegen time, we would like to basically cast this set type into a concrete implementation


@dataclass(kw_only=True, frozen=True)
class SetType(BaseType):
    """Representation of the Simile Set type.
    This class contains the interface of sets, but can be expanded."""

    element_type: BaseType  # We opt not for generic types since we dont want to hijack python's type system - we want to make our own
    """The Simile-type of elements in the set"""

    # These functions control the return types and trait-trait interactions (where applicable)
    # I suppose this kind-of simulates the program execution just looking at traits and element types

    # Type checking methods
    # TODO add classmethods like bag(), sequence(), relation() to create specific set subtypes easily, without needing to fiddle with traits
    def is_relation(self) -> bool:
        return isinstance(self.element_type, PairType)

    def is_sequence(self) -> bool:
        return isinstance(self.element_type, PairType) and self.element_type.left.eq_type(IntType())

    def is_bag(self) -> bool:
        return isinstance(self.element_type, PairType) and self.element_type.right.eq_type(IntType())

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, SetType):
            return False
        return self.element_type.eq_type(other.element_type, substitution_mapping)  # and self.relation_subtype == other.relation_subtype # TODO add the subtype check back in

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, SetType):
            return False
        return self.element_type.is_sub_type(other.element_type, substitution_mapping)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return SetType(element_type=self.element_type._replace_generic_types(lst), traits=self.traits)

    # Programming-oriented operations
    def copy(self) -> SetType:
        """Create a copy of the set."""
        return deepcopy(self)

    def clear(self) -> NoneType_:
        """Remove all elements from the set."""
        return NoneType_()

    def is_empty(self) -> BoolType:
        """Check if the set has no elements."""
        return BoolType()

    # Atomic operations
    def add(self, element: BaseType) -> NoneType_:
        """Add an element to the set."""
        return NoneType_()

    def remove(self, element: BaseType) -> NoneType_:
        """Remove an element from the set."""
        return NoneType_()

    def contains(self, element: BaseType) -> BoolType:
        """Check if an element is in the set (membership test)."""
        return BoolType()

    # Single operations
    def cardinality(self) -> IntType:
        """Return the number of elements in the set."""
        return IntType()

    def powerset(self) -> SetType:
        """Return the powerset of the set."""
        return SetType(element_type=self)

    def map(self, func: Callable[[BaseType], BaseType]) -> SetType:
        """Apply a function to each element in the set."""
        return SetType(element_type=func(self.element_type))

    def choice(self) -> BaseType:
        """Select an arbitrary element from the set."""
        return self.element_type

    def sum(self) -> BaseType:
        """Return the sum of all elements in the set."""
        return self.element_type

    def product(self) -> BaseType:
        """Return the product of all elements in the set."""
        return self.element_type

    def min(self) -> BaseType:
        """Return the minimum element in the set."""
        return self.element_type

    def max(self) -> BaseType:
        """Return the maximum element in the set."""
        return self.element_type

    def map_min(self, func: Callable[[BaseType], IntType]) -> BaseType:
        """Apply a weighting function to each element and return the minimum."""
        return self.element_type

    def map_max(self, func: Callable[[BaseType], IntType]) -> BaseType:
        """Apply a weighting function to each element and return the maximum."""
        return self.element_type

    # Binary operations
    def union(self, other: SetType) -> SetType:
        """Return the union of this set and another set."""
        return self

    def intersection(self, other: SetType) -> SetType:
        """Return the intersection of this set and another set."""
        return self

    def difference(self, other: SetType) -> SetType:
        """Return the difference of this set and another set."""
        return self

    def symmetric_difference(self, other: SetType) -> SetType:
        """Return the symmetric difference of this set and another set."""
        return self

    def cartesian_product(self, other: SetType) -> SetType:
        """Return the cartesian product of this set and another set."""
        return SetType(
            element_type=PairType(
                left=self.element_type,
                right=other.element_type,
            )
        )

    def is_subset(self, other: SetType) -> BoolType:
        """Check if this set is a subset of another set."""
        return BoolType()

    def is_disjoint(self, other: SetType) -> BoolType:
        """Check if this set and another set are disjoint."""
        return BoolType()

    def is_superset(self, other: SetType) -> BoolType:
        """Check if this set is a superset of another set."""
        return BoolType()
