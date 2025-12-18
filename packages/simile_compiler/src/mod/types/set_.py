from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable, Generic, TypeVar, Any, Iterable


T = TypeVar("T", bound="BaseType")
V = TypeVar("V", bound="BaseType")
E = TypeVar("E", bound="BaseType")
L = TypeVar("L", bound="BaseType")
R = TypeVar("R", bound="BaseType")


@dataclass
class Trait:
    """A trait that modifies the behavior of a type (usually based on the element type or expected element values).

    Traits can be used to indicate special properties of the set, such as ordering,
    uniqueness, or other characteristics that affect how the set operates.
    """

    _name: str = field(init=False)
    """Name of the trait"""


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


@dataclass(kw_only=True, frozen=True)
class NoneType_(BaseType):
    """Intended for statements without a type, not expressions. For example, a while loop node doesn't have a type."""


@dataclass(kw_only=True, frozen=True)
class BoolType(BaseType):
    pass


@dataclass(kw_only=True, frozen=True)
class StringType(BaseType):
    pass


@dataclass(kw_only=True, frozen=True)
class IntType(BaseType):
    pass


@dataclass(kw_only=True, frozen=True)
class FloatType(BaseType):
    pass


@dataclass(kw_only=True, frozen=True)
class Literal(BaseType):
    """A literal value of a specific type T."""

    value: BaseType


@dataclass(kw_only=True, frozen=True)
class GenericType(BaseType):
    """Generic types are used primarily for resolving generic procedures/functions into a specific type based on context.

    IDs are only locally valid (i.e., introduced by a procedure argument and used by a procedure's return value).
    """

    id_: str


@dataclass(kw_only=True, frozen=True)
class DeferToSymbolTable(BaseType):
    """Types dependent on this will not be resolved until the analysis phase"""

    lookup_type: str
    """Identifier to look up in table"""


@dataclass(kw_only=True, frozen=True)
class ModuleImports(BaseType):
    # import these objects into the module namespace
    import_objects: dict[str, BaseType]


@dataclass(kw_only=True, frozen=True)
class TupleType(BaseType):
    items: tuple[BaseType, ...]

    def __post__init__(self):
        for item in self.items:
            if not isinstance(item, BaseType):
                raise TypeError(f"TupleType items must be BaseType instances, got {type(item)}")


@dataclass(kw_only=True, frozen=True)
class PairType(TupleType):
    """Maplet type"""

    def __init__(self, left: BaseType, right: BaseType) -> None:
        super().__init__((left, right))  # type: ignore

    @property
    def left(self) -> BaseType:
        # python cant handle generic tuples just yet, so just ignore the type checker here
        return self.items[0]  # type: ignore

    @property
    def right(self) -> BaseType:
        return self.items[1]  # type: ignore


# TODO we basically need a SetSimulator that will return the expected type, element type, and traits when executing a set operation
# Then we need a code generator that will follow through on the simulator's typed promise - maybe make a mirror class that outputs generated code instead of types?
# Whats the cleanest way to do this?
#
# At codegen time, we would like to basically cast this set type into a concrete implementation


@dataclass(kw_only=True, frozen=True)
class SetType(BaseType):
    """Representation of the Simile Set type. Also intended to serve as a library of sorts for when codegen time rolls around.

    This class contains the interface of sets, but can be expanded.
    The engine contains the actual executable implementation of the set operations."""

    element_type: BaseType  # T is the python structure type, the value of T is the type the compiler actually cares about
    """The Simile-type of elements in the set"""
    traits: list[Trait] = field(default_factory=list)
    """Traits are subtypes that modify the behavior of the set (by changing the choice of engine)"""

    # These functions control the return types and trait-trait interactions (where applicable)
    # I suppose this kind-of simulates the program execution just looking at traits and element types

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
    def add(self, element: T) -> NoneType_:
        """Add an element to the set."""
        return NoneType_()

    def remove(self, element: T) -> NoneType_:
        """Remove an element from the set."""
        return NoneType_()

    def contains(self, element: T) -> BoolType:
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
        return SetType(element_type=PairType(self.element_type, other.element_type))

    def is_subset(self, other: SetType) -> BoolType:
        """Check if this set is a subset of another set."""
        return BoolType()

    def is_disjoint(self, other: SetType) -> BoolType:
        """Check if this set and another set are disjoint."""
        return BoolType()

    def is_superset(self, other: SetType) -> BoolType:
        """Check if this set is a superset of another set."""
        return BoolType()


@dataclass(kw_only=True, frozen=True)
class StructTypeDef(BaseType):
    # Internally a (many-to-one) (total on defined fields) function
    fields: dict[str, BaseType]


@dataclass(kw_only=True, frozen=True)
class EnumTypeDef(SetType):
    # Internally a set of identifiers
    element_type = StringType()  # TODO add trait domain
    members: set[str] = field(default_factory=set)


@dataclass(kw_only=True, frozen=True)
class ProcedureTypeDef(BaseType):
    arg_types: dict[str, BaseType]
    return_type: BaseType
