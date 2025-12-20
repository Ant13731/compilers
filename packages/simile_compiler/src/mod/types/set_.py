from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, TypeVar, Type

from src.mod.types.traits import Trait
from src.mod.types.base import BaseType, BoolType, AnyType_, SimileTypeError
from src.mod.types.primitive import NoneType_, IntType
from src.mod.types.tuple_ import PairType


# TODO we basically need a SetSimulator that will return the expected type, element type, and traits when executing a set operation
# Then we need a code generator that will follow through on the simulator's typed promise - maybe make a mirror class that outputs generated code instead of types?
# Whats the cleanest way to do this?
#
# At codegen time, we would like to basically cast this set type into a concrete implementation

T = TypeVar("T", bound="SetType")
V = TypeVar("V", bound="BaseType")


@dataclass(kw_only=True)
class SetType(BaseType):
    """Representation of the Simile Set type.
    This class contains the interface of sets, but can be expanded."""

    # We opt not for generic types since we dont want to hijack python's type system - we want to make our own
    _element_type: BaseType = field(init=False)
    """The Simile-type of elements in the set"""

    def __init__(self, *, element_type: BaseType, traits: list[Trait] | None = None) -> None:
        if traits is None:
            traits = []
        super().__init__(traits=traits)
        self._element_type = element_type

    @property
    def element_type(self) -> BaseType:
        return self._element_type

    # These functions control the return types and trait-trait interactions (where applicable)
    # I suppose this kind-of simulates the program execution just looking at traits and element types

    # Type checking methods
    # TODO add classmethods like bag(), sequence(), relation() to create specific set subtypes easily, without needing to fiddle with traits
    def is_relation(self) -> bool:
        return isinstance(self.element_type, PairType)

    def is_sequence(self) -> bool:
        return isinstance(self.element_type, PairType) and self.element_type.left.is_eq_type(IntType())

    def is_bag(self) -> bool:
        return isinstance(self.element_type, PairType) and self.element_type.right.is_eq_type(IntType())

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, SetType):
            return False
        return self.element_type.is_eq_type(other.element_type, substitution_mapping)

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

    def in_(self, element: BaseType) -> BoolType:
        """Check if an element is in the set (membership test)."""
        return BoolType()

    def not_in(self, element: BaseType) -> BoolType:
        """Check if an element is in the set (membership test)."""
        return self.in_(element).not_()

    @classmethod
    def enumeration(cls: Type[T], element_types: list[BaseType]) -> T:
        """Create a set from an enumeration of elements of a specific type."""
        if element_types == []:
            return cls(element_type=AnyType_())

        return cls(element_type=BaseType.max_type(element_types))

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
        # TODO Fail if empty trait is present
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
        # Need to call function to type check the function with the input type
        func(self.element_type)
        return self.element_type

    def map_max(self, func: Callable[[BaseType], IntType]) -> BaseType:
        """Apply a weighting function to each element and return the maximum."""
        func(self.element_type)
        return self.element_type

    # Binary operations
    def union(self, other: SetType) -> SetType:
        """Return the union of this set and another set."""
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform union on sets with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            # TODO this should check for subtypes?
            raise SimileTypeError(f"Cannot perform union on sets with different types: {self} and {other}")

        return self

    def intersection(self, other: SetType) -> SetType:
        """Return the intersection of this set and another set."""
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform intersection on sets with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            # TODO this should check for subtypes?
            raise SimileTypeError(f"Cannot perform intersection on sets with different types: {self} and {other}")
        return self

    def difference(self, other: SetType) -> SetType:
        """Return the difference of this set and another set."""
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform difference on sets with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            # TODO this should check for subtypes?
            raise SimileTypeError(f"Cannot perform difference on sets with different types: {self} and {other}")
        return self

    def symmetric_difference(self, other: SetType) -> SetType:
        """Return the symmetric difference of this set and another set."""
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform symmetric difference on sets with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            # TODO this should check for subtypes?
            raise SimileTypeError(f"Cannot perform symmetric difference on sets with different types: {self} and {other}")
        return self

    def cartesian_product(self, other: SetType) -> RelationType:
        """Return the cartesian product of this set and another set."""
        return RelationType(
            left=self.element_type,
            right=other.element_type,
        )

    def is_disjoint(self, other: SetType) -> BoolType:
        """Check if this set and another set are disjoint."""
        return BoolType()

    def is_subset(self, other: SetType) -> BoolType:
        """Check if this set is a subset of another set."""
        return BoolType()

    def is_subset_equals(self, other: SetType) -> BoolType:
        return BoolType()

    def is_superset(self, other: SetType) -> BoolType:
        """Check if this set is a superset of another set."""
        return BoolType()

    def is_superset_equals(self, other: SetType) -> BoolType:
        return BoolType()

    def not_is_subset(self, other: SetType) -> BoolType:
        return self.is_subset(other).not_()

    def not_is_subset_equals(self, other: SetType) -> BoolType:
        return self.is_subset_equals(other).not_()

    def not_is_superset(self, other: SetType) -> BoolType:
        return self.is_superset(other).not_()

    def not_is_superset_equals(self, other: SetType) -> BoolType:
        return self.is_superset_equals(other).not_()

    # N-ary operations


@dataclass(kw_only=True)
class RelationType(SetType):

    def __init__(self, *, left: BaseType, right: BaseType, traits: list[Trait] | None = None) -> None:
        if traits is None:
            traits = []
        super().__init__(element_type=PairType(left=left, right=right), traits=traits)

    @property
    def left(self) -> BaseType:
        assert isinstance(self.element_type, PairType)
        return self.element_type.left

    @property
    def right(self) -> BaseType:
        assert isinstance(self.element_type, PairType)
        return self.element_type.right

    def inverse(self) -> RelationType:
        return RelationType(left=self.right, right=self.left)

    def composition(self, other: RelationType) -> RelationType:
        if not self.right.is_eq_type(other.left):
            raise SimileTypeError(f"Cannot compose relations with incompatible types: {self} and {other}")
        return RelationType(left=self.left, right=other.right)

    def function_call(self, argument: BaseType) -> BaseType:
        if not argument.is_eq_type(self.left):
            raise SimileTypeError(f"Cannot call relation as function with incompatible argument type: {argument} for relation {self}")
        return self.right

    def image(self, argument: BaseType) -> SetType:
        if not argument.is_eq_type(self.left):
            raise SimileTypeError(f"Cannot call relation as function with incompatible argument type: {argument} for relation {self}")
        return SetType(element_type=self.right)

    def overriding(self, other: RelationType) -> RelationType:
        if not self.left.is_eq_type(other.left) or not self.right.is_eq_type(other.right):
            raise SimileTypeError(f"Cannot override relations with incompatible types: {self} and {other}")
        return self

    def domain(self) -> SetType:
        return SetType(element_type=self.left)

    def range_(self) -> SetType:
        return SetType(element_type=self.right)

    def domain_restriction(self, domain_set: SetType) -> RelationType:
        if not self.domain().is_eq_type(domain_set):
            raise SimileTypeError(f"Cannot perform domain restriction with incompatible set element type: {domain_set.element_type} for relation {self}")
        return self

    def domain_subtraction(self, domain_set: SetType) -> RelationType:
        if not self.domain().is_eq_type(domain_set):
            raise SimileTypeError(f"Cannot perform domain subtraction with incompatible set element type: {domain_set.element_type} for relation {self}")
        return self

    def range_restriction(self, range_set: SetType) -> RelationType:
        if not self.range_().is_eq_type(range_set):
            raise SimileTypeError(f"Cannot perform range restriction with incompatible set element type: {range_set.element_type} for relation {self}")
        return self

    def range_subtraction(self, range_set: SetType) -> RelationType:
        if not self.range_().is_eq_type(range_set):
            raise SimileTypeError(f"Cannot perform range subtraction with incompatible set element type: {range_set.element_type} for relation {self}")
        return self

    def bag_image(self, bag: BagType) -> BagType:
        # Get traits from here. This also needs to be run to check for type errors from dependent operations
        self.inverse().composition(bag)

        return BagType(element_type=self.right)


@dataclass(kw_only=True)
class BagType(RelationType):

    def __init__(self, *, element_type: BaseType, traits: list[Trait] | None = None) -> None:
        if traits is None:
            traits = []
        super().__init__(left=element_type, right=IntType(), traits=traits)

    @property
    def element_type(self) -> BaseType:
        return self.left

    def bag_union(self, other: BagType) -> BagType:
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform union on bags with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            raise SimileTypeError(f"Cannot perform union on bags with different types: {self} and {other}")
        return self

    def bag_intersection(self, other: BagType) -> BagType:
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform intersection on bags with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            raise SimileTypeError(f"Cannot perform intersection on bags with different types: {self} and {other}")
        return self

    def bag_add(self, other: BagType) -> BagType:
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform addition on bags with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            raise SimileTypeError(f"Cannot perform addition on bags with different types: {self} and {other}")
        return self

    def bag_difference(self, other: BagType) -> BagType:
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot perform difference on bags with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            raise SimileTypeError(f"Cannot perform difference on bags with different types: {self} and {other}")
        return self

    def size(self) -> IntType:
        """Return the total number of elements in the bag, counting multiplicities."""
        return IntType()


@dataclass(kw_only=True)
class SequenceType(RelationType):

    def __init__(self, *, element_type: BaseType, traits: list[Trait] | None = None) -> None:
        if traits is None:
            traits = []
        super().__init__(left=IntType(), right=element_type, traits=traits)

    @property
    def element_type(self) -> BaseType:
        return self.right

    def concat(self, other: SequenceType) -> SequenceType:
        if not self.element_type.is_eq_type(other.element_type):
            raise SimileTypeError(f"Cannot concatenate sequences with different element types: {self.element_type} and {other.element_type}")

        if not self.is_eq_type(other):
            # TODO this should check for subtypes?
            raise SimileTypeError(f"Cannot concatenate sequences with different types: {self} and {other}")
        return self
