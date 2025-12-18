from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar, Any, Iterable

from src.mod.ast_.symbol_table_types import BaseSimileType, PairType

T = TypeVar("T")
V = TypeVar("V")
E = TypeVar("E")


@dataclass
class SetImplementationCodeGenerator:
    """The engine that implements the set operations.

    This is a placeholder for the actual implementation of set operations.
    The engine can be expanded to include various data structures and algorithms
    for different set behaviors (e.g., ordered sets, unordered sets, etc.)."""

    _name: str = field(init=False)
    """Name of the engine implementation"""

    _propose_change_to_engine_type: type["SetImplementationCodeGenerator"] | None = None
    """If the engine determines that a different engine type would be more efficient, it can propose a change to the set interface.

    The actual change must be handled by the Set class."""

    # Each of these returns implementation code in the form of a string
    def add(self, element: Any) -> str:
        raise NotImplementedError

    def remove(self, element: Any) -> str:
        raise NotImplementedError

    def copy(self) -> str:
        raise NotImplementedError

    def clear(self) -> str:
        raise NotImplementedError

    def is_empty(self) -> str:
        raise NotImplementedError

    def contains(self, element: Any) -> str:
        raise NotImplementedError

    def from_collection(self, collection: Iterable[Any]) -> str:
        raise NotImplementedError

    def cardinality(self) -> str:
        raise NotImplementedError

    def powerset(self) -> str:
        raise NotImplementedError

    def map(self, func) -> str:
        raise NotImplementedError

    def choice(self) -> str:
        raise NotImplementedError

    def sum(self) -> str:
        raise NotImplementedError

    def product(self) -> str:
        raise NotImplementedError

    def min(self) -> str:
        raise NotImplementedError

    def max(self) -> str:
        raise NotImplementedError

    def map_min(self, func) -> str:
        raise NotImplementedError

    def map_max(self, func) -> str:
        raise NotImplementedError

    def union(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def intersection(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def difference(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def symmetric_difference(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def cartesian_product(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def is_subset(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def is_disjoint(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def is_superset(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError

    def equal(self, other: "SetImplementationCodeGenerator") -> str:
        raise NotImplementedError


@dataclass
class Trait:
    """A trait that modifies the behavior of a type (usually based on the element type or expected element values).

    Traits can be used to indicate special properties of the set, such as ordering,
    uniqueness, or other characteristics that affect how the set operates.
    """

    _name: str = field(init=False)
    """Name of the trait"""


@dataclass
class Set(Generic[T]):
    """Library for Simile Set types.

    This class contains the interface of sets, but can be expanded.
    The engine contains the actual executable implementation of the set operations."""

    element_type: T  # T is the python structure type, the value of T is the type the compiler actually cares about
    """The Simile-type of elements in the set"""
    traits: list[Trait] = field(default_factory=list)
    """Traits are subtypes that modify the behavior of the set (by changing the choice of engine)"""

    implementation_codegen_override: type[SetImplementationCodeGenerator] | None = None
    """If provided, this engine type will be used instead of the one chosen by the traits and element type."""

    _implementation_code_generator: SetImplementationCodeGenerator = field(init=False)
    """The underlying data structure and operation implementations"""

    def __post_init__(self):
        if self.implementation_codegen_override is not None:
            self._implementation_code_generator = self.implementation_codegen_override()
        else:
            self._implementation_code_generator = self._choose_engine(self.element_type, self.traits)

    @staticmethod
    def _choose_engine(element_type: T, traits: list[Trait]) -> SetImplementationCodeGenerator:
        """One function determines the engine based on element type and traits - this should be the only
        function making decisions based on trait and element type information.

        Such decisions are made every time the engine is formed (when an engine changes, the underlying data structure changes)
        """
        # if Trait.Ordered in traits:
        #     return SetEngine.OrderedSet
        # else:
        #     return SetEngine.UnorderedSet
        raise NotImplementedError

    # These functions control the return types and trait-trait interactions (where applicable)
    # I suppose this kind-of simulates the program execution just looking at traits and element types

    # Atomic operations
    def add(self, element: T) -> BaseSimileType.None_:
        """Add an element to the set."""
        self._implementation_code_generator.add(element)

    def remove(self, element: T) -> BaseSimileType.None_:
        """Remove an element from the set."""
        self._implementation_code_generator.remove(element)

    def copy(self) -> "Set[T]":
        """Create a copy of the set."""
        new_set = Set(
            element_type=self.element_type,
            traits=self.traits.copy(),
            implementation_codegen_override=self.implementation_codegen_override,
        )
        new_set._implementation_code_generator = self._implementation_code_generator.copy()
        return new_set

    def clear(self) -> BaseSimileType.None_:
        """Remove all elements from the set."""
        self._implementation_code_generator.clear()

    def cast(self, caster: "Callable[[Set[T]], V]") -> V:
        """Cast the entire set to a different type."""
        raise NotImplementedError

    def is_empty(self) -> BaseSimileType.Bool:
        """Check if the set has no elements."""
        return self._implementation_code_generator.is_empty()

    @classmethod
    def from_collection(
        cls,
        collection: Iterable[T],
        element_type: T,
        traits: list[Trait] | None = None,
    ) -> "Set[T]":
        """Create a set from a collection (e.g., list, tuple)."""
        new_set = cls(element_type=element_type, traits=traits or [])
        new_set._implementation_code_generator.from_collection(collection)
        return new_set

    def contains(self, element: T) -> BaseSimileType.Bool:
        """Check if an element is in the set (membership test)."""
        return self._implementation_code_generator.contains(element)

    # Single operations
    def cardinality(self) -> BaseSimileType.Int:
        """Return the number of elements in the set."""
        return self._implementation_code_generator.cardinality()

    def powerset(self) -> "Set[Set[T]]":
        """Return the powerset of the set."""
        return self._implementation_code_generator.powerset()

    def map(self, func: Callable[[T], V]) -> "Set[V]":
        """Apply a function to each element in the set."""
        return self._implementation_code_generator.map(func)

    def choice(self) -> T:
        """Select an arbitrary element from the set."""
        return self._implementation_code_generator.choice()

    def sum(self) -> T:
        """Return the sum of all elements in the set."""
        return self._implementation_code_generator.sum()

    def product(self) -> T:
        """Return the product of all elements in the set."""
        return self._implementation_code_generator.product()

    def min(self) -> T:
        """Return the minimum element in the set."""
        return self._implementation_code_generator.min()

    def max(self) -> T:
        """Return the maximum element in the set."""
        return self._implementation_code_generator.max()

    def map_min(self, func: Callable[[T], BaseSimileType.Int]) -> T:
        """Apply a weighting function to each element and return the minimum."""
        return self._implementation_code_generator.map_min(func)

    def map_max(self, func: Callable[[T], BaseSimileType.Int]) -> T:
        """Apply a weighting function to each element and return the maximum."""
        return self._implementation_code_generator.map_max(func)

    # Binary operations
    def union(self, other: "Set[T]") -> "Set[T]":
        """Return the union of this set and another set."""
        return self._implementation_code_generator.union(other)

    def intersection(self, other: "Set[T]") -> "Set[T]":
        """Return the intersection of this set and another set."""
        return self._implementation_code_generator.intersection(other)

    def difference(self, other: "Set[T]") -> "Set[T]":
        """Return the difference of this set and another set."""
        return self._implementation_code_generator.difference(other)

    def symmetric_difference(self, other: "Set[T]") -> "Set[T]":
        """Return the symmetric difference of this set and another set."""
        return self._implementation_code_generator.symmetric_difference(other)

    def cartesian_product(self, other: "Set[V]") -> "Set[PairType[T, V]]":
        """Return the cartesian product of this set and another set."""
        return self._implementation_code_generator.cartesian_product(other)

    def is_subset(self, other: "Set[T]") -> BaseSimileType.Bool:
        """Check if this set is a subset of another set."""
        return self._implementation_code_generator.is_subset(other)

    def is_disjoint(self, other: "Set[T]") -> BaseSimileType.Bool:
        """Check if this set and another set are disjoint."""
        return self._implementation_code_generator.is_disjoint(other)

    def is_superset(self, other: "Set[T]") -> BaseSimileType.Bool:
        """Check if this set is a superset of another set."""
        return self._implementation_code_generator.is_superset(other)

    def equal(self, other: "Set[T]") -> BaseSimileType.Bool:
        """Check if this set is equal to another set."""
        return self._implementation_code_generator.equal(other)
