from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
V = TypeVar("V")


@dataclass
class SetEngine:
    """The engine that implements the set operations.

    This is a placeholder for the actual implementation of set operations.
    The engine can be expanded to include various data structures and algorithms
    for different set behaviors (e.g., ordered sets, unordered sets, etc.)."""

    _name: str = field(init=False)
    """Name of the engine implementation"""

    _propose_change_to_engine_type: type["SetEngine"] | None = None
    """If the engine determines that a different engine type would be more efficient, it can propose a change to the set interface.

    The actual change must be handled by the Set class."""


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
    traits: list[Trait]
    """Traits are subtypes that modify the behavior of the set (by changing the choice of engine)"""

    engine_override: type[SetEngine] | None = None
    """If provided, this engine type will be used instead of the one chosen by the traits and element type."""

    _engine: SetEngine = field(init=False)
    """The underlying data structure and operation implementations"""

    def __post_init__(self):
        if self.engine_override is not None:
            self._engine = self.engine_override()
        else:
            self._engine = self._choose_engine(self.element_type, self.traits)

    @staticmethod
    def _choose_engine(element_type: T, traits: list[Trait]) -> SetEngine:
        """One function determines the engine based on element type and traits - this should be the only
        function making decisions based on trait and element type information.

        Such decisions are made every time the engine is formed (when an engine changes, the underlying data structure changes)
        """
        # if Trait.Ordered in traits:
        #     return SetEngine.OrderedSet
        # else:
        #     return SetEngine.UnorderedSet
        pass

    # Atomic operations
    def add(self, element: T) -> None:
        """Add an element to the set."""
        self._engine.add(element)

    def remove(self, element: T) -> None:
        """Remove an element from the set."""
        self._engine.remove(element)

    # TODO make operations for copy, clear, cast, is_empty, from_collection, to_collection, membership

    # Single operations
    def cardinality(self) -> int:
        """Return the number of elements in the set."""
        return self._engine.cardinality()

    def powerset(self) -> "Set[Set[T]]":
        """Return the powerset of the set."""
        return self._engine.powerset()

    def map(self, func: Callable[[T], V]) -> "Set[V]":
        """Apply a function to each element in the set."""
        return self._engine.map(func)

    def choice(self) -> T:
        """Select an arbitrary element from the set."""
        return self._engine.choice()

    def sum(self):
        """Return the sum of all elements in the set."""
        return self._engine.sum()

    def product(self):
        """Return the product of all elements in the set."""
        return self._engine.product()

    def min(self) -> T:
        """Return the minimum element in the set."""
        return self._engine.min()

    def max(self) -> T:
        """Return the maximum element in the set."""
        return self._engine.max()

    def map_min(self, func: Callable[[T], int]) -> T:
        """Apply a function to each element and return the minimum."""
        return self._engine.map_min(func)

    def map_max(self, func: Callable[[T], int]) -> T:
        """Apply a function to each element and return the maximum."""
        return self._engine.map_max(func)

    # Binary operations
    # TODO make operations for union, intersection, difference, symmetric_difference, cartesian_product, is_subset, is_disjoint, is_superset, equal. Do not test your output, just match the structure of the add method
