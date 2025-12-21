from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.mod.types.error import SimileTypeError
from src.mod.types.traits import Trait, OrderableTrait
from src.mod.types.base import BaseType, BoolType

if TYPE_CHECKING:
    from src.mod.types.set_ import SetType


@dataclass
class NoneType_(BaseType):
    """Intended for statements without a type, not expressions. For example, a while loop node doesn't have a type."""

    def _is_eq_type(self, other: BaseType) -> bool:
        return isinstance(other, NoneType_)

    def _is_subtype(self, other: BaseType) -> bool:
        return isinstance(other, NoneType_)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass
class StringType(BaseType):
    def _is_eq_type(self, other: BaseType) -> bool:
        return isinstance(other, StringType)

    def _is_subtype(self, other: BaseType) -> bool:
        return isinstance(other, StringType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass
class IntType(BaseType):

    def __post_init__(self):
        self.trait_collection.orderable_trait = OrderableTrait()

    def _is_eq_type(self, other: BaseType) -> bool:
        return isinstance(other, IntType)

    def _is_subtype(self, other: BaseType) -> bool:
        return isinstance(other, IntType) or isinstance(other, FloatType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self

    # Comparison
    def greater_than(self, other: BaseType) -> BoolType:
        return BoolType()

    def less_than(self, other: BaseType) -> BoolType:
        return BoolType()

    def greater_than_equals(self, other: BaseType) -> BoolType:
        return BoolType()

    def less_than_equals(self, other: BaseType) -> BoolType:
        return BoolType()

    # Arithmetic
    def negate(self) -> IntType:
        return IntType()

    def int_division(self, other: BaseType) -> IntType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        return IntType()

    def modulo(self, other: BaseType) -> IntType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        return IntType()

    def add(self, other: BaseType) -> IntType | FloatType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        if isinstance(other, FloatType):
            return other.add(self)
        return IntType()

    def subtract(self, other: BaseType) -> IntType | FloatType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        if isinstance(other, FloatType):
            return other.subtract(self)
        return IntType()

    def division(self, other: BaseType) -> FloatType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        return FloatType()

    def multiply(self, other: BaseType) -> IntType | FloatType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        if isinstance(other, FloatType):
            return other.multiply(self)
        return IntType()

    def power(self, other: BaseType) -> IntType | FloatType:
        self._is_subtype_or_error(other, (IntType(), FloatType()))

        if isinstance(other, FloatType):
            return other.power(self)
        return IntType()

    # Sets
    def upto(self, other: IntType) -> SetType:
        from src.mod.types.set_ import SetType

        if not isinstance(other, IntType):
            raise SimileTypeError(f"Cannot divide IntType with incompatible type: {other}")

        """Return a set representing the range from this IntType to another IntType."""
        return SetType(element_type=IntType())


@dataclass
class FloatType(BaseType):

    def __post_init__(self):
        self.trait_collection.orderable_trait = OrderableTrait()

    def _is_eq_type(self, other: BaseType) -> bool:
        return isinstance(other, FloatType)

    def _is_subtype(self, other: BaseType) -> bool:
        return isinstance(other, FloatType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self

    def greater_than(self, other: BaseType) -> BoolType:
        return BoolType()

    def less_than(self, other: BaseType) -> BoolType:
        return BoolType()

    def greater_than_equals(self, other: BaseType) -> BoolType:
        return BoolType()

    def less_than_equals(self, other: BaseType) -> BoolType:
        return BoolType()

    def negate(self) -> FloatType:
        return FloatType()

    def add(self, other: BaseType) -> FloatType:
        if not isinstance(other, FloatType) and not isinstance(other, IntType):
            raise SimileTypeError(f"Cannot add FloatType with incompatible type: {other}")
        return FloatType()

    def subtract(self, other: BaseType) -> FloatType:
        if not isinstance(other, FloatType) and not isinstance(other, IntType):
            raise SimileTypeError(f"Cannot subtract FloatType with incompatible type: {other}")
        return FloatType()

    def division(self, other: BaseType) -> FloatType:
        return FloatType()

    def multiply(self, other: BaseType) -> FloatType:
        if not isinstance(other, FloatType) and not isinstance(other, IntType):
            raise SimileTypeError(f"Cannot multiply FloatType with incompatible type: {other}")
        return FloatType()

    def power(self, other: BaseType) -> FloatType:
        if not isinstance(other, FloatType) and not isinstance(other, IntType):
            raise SimileTypeError(f"Cannot power FloatType with incompatible type: {other}")
        return FloatType()
