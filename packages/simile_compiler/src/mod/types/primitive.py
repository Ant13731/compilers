from __future__ import annotations
from dataclasses import dataclass

from src.mod.types.traits import Trait
from src.mod.types.base import BaseType


@dataclass(kw_only=True, frozen=True)
class NoneType_(BaseType):
    """Intended for statements without a type, not expressions. For example, a while loop node doesn't have a type."""

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, NoneType_)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, NoneType_)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass(kw_only=True, frozen=True)
class StringType(BaseType):
    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, StringType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, StringType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass(kw_only=True, frozen=True)
class IntType(BaseType):
    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, IntType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, IntType) or isinstance(other, FloatType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self


@dataclass(kw_only=True, frozen=True)
class FloatType(BaseType):
    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, FloatType)

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        return isinstance(other, FloatType)

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return self
