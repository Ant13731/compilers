import pytest
from dataclasses import dataclass, field

from src.mod.types import set_


@dataclass
class MockSetEngine(set_.SetEngine):
    """A simple mock engine for testing purposes."""

    _data: set = field(default_factory=set)
    _name: str = field(init=False, default="MockSetEngine")

    def add(self, element) -> None:
        """Add an element to the set."""
        self._data.add(element)

    def remove(self, element) -> None:
        """Remove an element from the set."""
        self._data.discard(element)

    def copy(self) -> "MockSetEngine":
        """Create a copy of the set engine."""
        new_engine = MockSetEngine()
        new_engine._data = self._data.copy()
        return new_engine

    def clear(self) -> None:
        """Remove all elements from the set."""
        self._data.clear()

    def is_empty(self) -> bool:
        """Check if the set has no elements."""
        return len(self._data) == 0

    def contains(self, element) -> bool:
        """Check if an element is in the set."""
        return element in self._data

    def from_collection(self, collection) -> None:
        """Populate the set from a collection."""
        self._data.clear()
        for item in collection:
            self._data.add(item)

    def to_collection(self) -> list:
        """Convert the set to a list."""
        return list(self._data)


class TestSetOperations:
    """Test the set operations implementation."""

    def test_copy(self):
        """Test that copy creates an independent copy of the set."""
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        s.add(1)
        s.add(2)
        s.add(3)

        s_copy = s.copy()
        
        # Verify the copy has the same elements
        assert s_copy.contains(1)
        assert s_copy.contains(2)
        assert s_copy.contains(3)
        
        # Verify modifications to copy don't affect original
        s_copy.add(4)
        assert s_copy.contains(4)
        assert not s.contains(4)

    def test_clear(self):
        """Test that clear removes all elements from the set."""
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        s.add(1)
        s.add(2)
        s.add(3)

        assert not s.is_empty()
        s.clear()
        assert s.is_empty()

    def test_is_empty(self):
        """Test is_empty returns correct values."""
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        
        assert s.is_empty()
        
        s.add(1)
        assert not s.is_empty()
        
        s.remove(1)
        assert s.is_empty()

    def test_contains(self):
        """Test membership testing."""
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        s.add(1)
        s.add(2)
        
        assert s.contains(1)
        assert s.contains(2)
        assert not s.contains(3)
        
        # Test __contains__ method (in operator)
        assert 1 in s
        assert 2 in s
        assert 3 not in s

    def test_from_collection(self):
        """Test creating a set from a collection."""
        collection = [1, 2, 3, 4, 5]
        # Create a set using from_collection class method
        # Note: we need to manually set up the engine since _choose_engine returns None
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        s._engine.from_collection(collection)
        
        assert s.contains(1)
        assert s.contains(2)
        assert s.contains(3)
        assert s.contains(4)
        assert s.contains(5)
        assert not s.contains(6)

    def test_to_collection(self):
        """Test converting a set to a collection."""
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        s.add(1)
        s.add(2)
        s.add(3)
        
        collection = s.to_collection()
        
        # Check that all elements are in the collection
        assert 1 in collection
        assert 2 in collection
        assert 3 in collection
        assert len(collection) == 3

    def test_cast(self):
        """Test casting a set to a different element type."""
        s = set_.Set(element_type=int, traits=[], engine_override=MockSetEngine)
        s.add(1)
        s.add(2)
        s.add(3)
        
        # Cast to string type (conceptual - the actual type conversion isn't performed)
        s_cast = s.cast(str)
        
        # Verify new set has different element type
        assert s_cast.element_type == str
        assert s.element_type == int
        
        # Verify elements were copied
        assert s_cast.contains(1)
        assert s_cast.contains(2)
        assert s_cast.contains(3)
