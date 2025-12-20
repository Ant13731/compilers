from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.mod.types.traits import Trait
from src.mod.types.base import BaseType, BoolType, SimileTypeError

if TYPE_CHECKING:
    from src.mod.types.set_ import SetType


@dataclass(kw_only=True)
class NoneType_(BaseType):
    """Intended for statements without a type, not expressions. For example, a while loop node doesn't have a type."""

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, NoneType_)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, NoneType_)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass(kw_only=True)
class StringType(BaseType):
    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, StringType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, StringType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass(kw_only=True)
class IntType(BaseType):
    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, IntType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
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
        return IntType()

    def modulo(self, other: BaseType) -> IntType:
        return IntType()

    def add(self, other: BaseType) -> IntType | FloatType:
        if isinstance(other, FloatType):
            return other.add(self)
        return IntType()

    def subtract(self, other: BaseType) -> IntType | FloatType:
        if isinstance(other, FloatType):
            return other.subtract(self)
        return IntType()

    def division(self, other: BaseType) -> FloatType:
        return FloatType()

    def multiply(self, other: BaseType) -> IntType | FloatType:
        if isinstance(other, FloatType):
            return other.multiply(self)
        return IntType()

    def power(self, other: BaseType) -> IntType | FloatType:
        if isinstance(other, FloatType):
            return other.power(self)
        return IntType()

    # Sets
    def upto(self, other: IntType) -> SetType:
        from src.mod.types.set_ import SetType

        """Return a set representing the range from this IntType to another IntType."""
        return SetType(element_type=IntType())


@dataclass(kw_only=True)
class FloatType(BaseType):
    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, FloatType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
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
