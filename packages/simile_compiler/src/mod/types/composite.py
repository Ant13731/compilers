from __future__ import annotations
from dataclasses import dataclass, field

from src.mod.types.traits import Trait
from src.mod.types.set_ import SetType
from src.mod.types.base import BaseType
from src.mod.types.primitive import StringType


@dataclass(kw_only=True, frozen=True)
class StructTypeDef(BaseType):
    # Internally a (many-to-one) (total on defined fields) function
    fields: dict[str, BaseType]

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, StructTypeDef):
            return False
        return all(
            f._eq_type(o, substitution_mapping)
            for f, o in zip(
                self.fields.values(),
                other.fields.values(),
            )
        )

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return StructTypeDef(fields={name: field._replace_generic_types(lst) for name, field in self.fields.items()}, traits=self.traits)


@dataclass(kw_only=True, frozen=True)
class EnumTypeDef(SetType):
    # Internally a set of identifiers
    element_type = StringType()  # TODO add trait domain
    members: set[str] = field(default_factory=set)

    # TODO add trait-domain subtyping and eq typing - maybe even in non-trait checked instances?


@dataclass(kw_only=True, frozen=True)
class ProcedureTypeDef(BaseType):
    arg_types: dict[str, BaseType]
    return_type: BaseType

    def _eq_type(self, other: BaseType, substitution_mapping: dict[str, BaseType]) -> bool:
        if not isinstance(other, ProcedureTypeDef):
            return False
        return all(
            f.eq_type(o, substitution_mapping)
            for f, o in zip(
                self.arg_types.values(),
                other.arg_types.values(),
            )
        ) and self.return_type.eq_type(
            other.return_type,
            substitution_mapping,
        )

    def _replace_generic_types(self, lst: list[BaseType]) -> BaseType:
        return ProcedureTypeDef(
            arg_types={name: arg_type._replace_generic_types(lst) for name, arg_type in self.arg_types.items()},
            return_type=self.return_type._replace_generic_types(lst),
            traits=self.traits,
        )
