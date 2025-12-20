from __future__ import annotations
from dataclasses import dataclass, field

from src.mod.types.traits import Trait
from src.mod.types.set_ import SetType
from src.mod.types.base import BaseType
from src.mod.types.primitive import StringType


@dataclass(kw_only=True)
class StructTypeDef(BaseType):
    # Internally a (many-to-one) (total on defined fields) function
    fields: dict[str, BaseType]

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, StructTypeDef):
            return False
        return all(
            f._is_eq_type(o, substitution_mapping)
            for f, o in zip(
                self.fields.values(),
                other.fields.values(),
            )
        )

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, StructTypeDef):
            return False

        for name in other.fields:
            # Records are only subtypes if they have at least all the same fields as other
            if name not in self.fields:
                return False

            # Fields that do match must be subtypes
            if not self.fields[name].is_sub_type(other.fields[name], substitution_mapping):
                return False

        return True

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return StructTypeDef(fields={name: field._replace_generic_types(lst) for name, field in self.fields.items()}, traits=self.traits)


@dataclass(kw_only=True)
class EnumTypeDef(SetType):
    # Internally a set of identifiers
    # element_type = StringType()  # TODO add trait domain
    members: set[str] = field(default_factory=set)

    def __post_init__(self):
        self._element_type = StringType()

    # TODO add trait-domain subtyping and eq typing - maybe even in non-trait checked instances?


@dataclass(kw_only=True)
class ProcedureTypeDef(BaseType):
    arg_types: dict[str, BaseType]
    return_type: BaseType

    def _is_eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, ProcedureTypeDef):
            return False
        return all(
            self_arg.is_eq_type(other_arg, substitution_mapping)
            for self_arg, other_arg in zip(
                self.arg_types.values(),
                other.arg_types.values(),
            )
        ) and self.return_type.is_eq_type(
            other.return_type,
            substitution_mapping,
        )

    def _is_sub_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, ProcedureTypeDef):
            return False
        return all(
            other_arg.is_eq_type(self_arg, substitution_mapping)
            for self_arg, other_arg in zip(
                self.arg_types.values(),
                other.arg_types.values(),
            )
        ) and self.return_type.is_sub_type(
            other.return_type,
            substitution_mapping,
        )

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return ProcedureTypeDef(
            arg_types={name: arg_type._replace_generic_types(lst) for name, arg_type in self.arg_types.items()},
            return_type=self.return_type._replace_generic_types(lst),
            traits=self.traits,
        )
