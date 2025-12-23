from __future__ import annotations
from dataclasses import dataclass, field
from collections import OrderedDict

from src.mod.types.error import SimileTypeError
from src.mod.types.traits import Trait, DomainTrait, LiteralTrait, ImmutableTrait
from src.mod.types.set_ import SetType
from src.mod.types.base import BaseType
from src.mod.types.primitive import StringType


@dataclass
class RecordType(BaseType):
    # Internally a (many-to-one) (total on defined fields) function
    fields: OrderedDict[str, BaseType]

    def _is_eq_type(self, other: BaseType) -> bool:
        if not isinstance(other, RecordType):
            return False
        return all(
            self_field._is_eq_type(other_field)
            for self_field, other_field in zip(
                self.fields.values(),
                other.fields.values(),
            )
        )

    def _is_subtype(self, other: BaseType) -> bool:
        if not isinstance(other, RecordType):
            return False

        for name in other.fields:
            # Records are only subtypes if they have at least all the same fields as other
            if name not in self.fields:
                return False

            # Fields that do match must be subtypes
            if not self.fields[name].is_subtype(other.fields[name]):
                return False

        return True

    def access(self, field_name: str) -> BaseType:
        if field_name not in self.fields:
            raise KeyError(f"RecordType has no field named '{field_name}'")
        return self.fields[field_name]


@dataclass
class ProcedureType(BaseType):
    arg_types: OrderedDict[str, BaseType]
    return_type: BaseType

    def _is_eq_type(self, other: BaseType) -> bool:
        if not isinstance(other, ProcedureType):
            return False
        return all(
            self_arg.is_eq_type(other_arg)
            for self_arg, other_arg in zip(
                self.arg_types.values(),
                other.arg_types.values(),
            )
        ) and self.return_type.is_eq_type(
            other.return_type,
        )

    def _is_subtype(self, other: BaseType) -> bool:
        if not isinstance(other, ProcedureType):
            return False
        return all(
            other_arg.is_eq_type(self_arg)
            for self_arg, other_arg in zip(
                self.arg_types.values(),
                other.arg_types.values(),
            )
        ) and self.return_type.is_subtype(
            other.return_type,
        )

    def call(self, arg_types: list[BaseType]) -> BaseType:
        if len(arg_types) != len(self.arg_types):
            raise SimileTypeError(f"Procedure called with incorrect number of arguments. Expected {len(self.arg_types)}, got {len(arg_types)}")

        for provided_type, (arg_name, expected_type) in zip(arg_types, self.arg_types.items()):
            if not provided_type.is_subtype(expected_type):
                raise SimileTypeError(f"Procedure argument '{arg_name}' expected type {expected_type}, got {provided_type}")
        # TODO check for generics here - ex, if return type is generic, look for the actual type in one of its arguments
        # generic ids should match
        # TODO also need to account for the return type matching the type of the returned value

        return self.return_type
