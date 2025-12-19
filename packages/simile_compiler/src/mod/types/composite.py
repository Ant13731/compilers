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


@dataclass(kw_only=True, frozen=True)
class EnumTypeDef(SetType):
    # Internally a set of identifiers
    element_type = StringType()  # TODO add trait domain
    members: set[str] = field(default_factory=set)


@dataclass(kw_only=True, frozen=True)
class ProcedureTypeDef(BaseType):
    arg_types: dict[str, BaseType]
    return_type: BaseType
