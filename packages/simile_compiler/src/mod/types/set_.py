from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any, Iterable

T = TypeVar("T")
U = TypeVar("U")


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

    def add(self, element: Any) -> None:
        """Add an element to the set."""
        raise NotImplementedError("Subclasses must implement add")

    def remove(self, element: Any) -> None:
        """Remove an element from the set."""
        raise NotImplementedError("Subclasses must implement remove")

    def copy(self) -> "SetEngine":
        """Create a copy of the set engine."""
        raise NotImplementedError("Subclasses must implement copy")

    def clear(self) -> None:
        """Remove all elements from the set."""
        raise NotImplementedError("Subclasses must implement clear")

    def is_empty(self) -> bool:
        """Check if the set has no elements."""
        raise NotImplementedError("Subclasses must implement is_empty")

    def contains(self, element: Any) -> bool:
        """Check if an element is in the set (membership test)."""
        raise NotImplementedError("Subclasses must implement contains")

    def from_collection(self, collection: Iterable[Any]) -> None:
        """Populate the set from a collection."""
        raise NotImplementedError("Subclasses must implement from_collection")

    def to_collection(self) -> list[Any]:
        """Convert the set to a collection (list)."""
        raise NotImplementedError("Subclasses must implement to_collection")


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

    def copy(self) -> "Set[T]":
        """Create a copy of the set."""
        new_set = Set(
            element_type=self.element_type,
            traits=self.traits.copy(),
            engine_override=self.engine_override
        )
        new_set._engine = self._engine.copy()
        return new_set

    def clear(self) -> None:
        """Remove all elements from the set."""
        self._engine.clear()

    def cast(self, new_element_type: U) -> "Set[U]":
        """Cast the set to a different element type.
        
        This creates a new set with a different element type, preserving the traits
        and potentially changing the engine based on the new type.
        """
        new_set = Set(
            element_type=new_element_type,
            traits=self.traits.copy(),
            engine_override=self.engine_override
        )
        # Copy the elements through collection conversion
        elements = self._engine.to_collection()
        new_set._engine.from_collection(elements)
        return new_set

    def is_empty(self) -> bool:
        """Check if the set has no elements."""
        return self._engine.is_empty()

    @classmethod
    def from_collection(cls, collection: Iterable[T], element_type: T, traits: list[Trait] | None = None) -> "Set[T]":
        """Create a set from a collection (e.g., list, tuple).
        
        Args:
            collection: An iterable containing elements to populate the set
            element_type: The type of elements in the set
            traits: Optional list of traits to apply to the set
        """
        if traits is None:
            traits = []
        new_set = cls(element_type=element_type, traits=traits)
        new_set._engine.from_collection(collection)
        return new_set

    def to_collection(self) -> list[T]:
        """Convert the set to a collection (list)."""
        return self._engine.to_collection()

    def contains(self, element: T) -> bool:
        """Check if an element is in the set (membership test)."""
        return self._engine.contains(element)

    def __contains__(self, element: T) -> bool:
        """Support the 'in' operator for membership testing."""
        return self.contains(element)

    # Single operations
    # TODO make operations for cardinality, powerset, map, choice, sum, product, min, max, map_min, map_max

    # Binary operations
    # TODO make operations for union, intersection, difference, symmetric_difference, cartesian_product, is_subset, is_disjoint, is_superset, equal
