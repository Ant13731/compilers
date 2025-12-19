from __future__ import annotations
from dataclasses import dataclass

from src.mod.types.base import BaseType
from src.mod.types.traits import Trait


@dataclass(kw_only=True, frozen=True)
class Literal(BaseType):
    """A literal value of a specific type T."""

    value: BaseType


@dataclass(kw_only=True, frozen=True)
class GenericType(BaseType):
    """Generic types are used primarily for resolving generic procedures/functions into a specific type based on context.

    IDs are only locally valid (i.e., introduced by a procedure argument and used by a procedure's return value).
    """

    id_: str


@dataclass(kw_only=True, frozen=True)
class DeferToSymbolTable(BaseType):
    """Types dependent on this will not be resolved until the analysis phase"""

    lookup_type: str
    """Identifier to look up in table"""


@dataclass(kw_only=True, frozen=True)
class ModuleImports(BaseType):
    # import these objects into the module namespace
    import_objects: dict[str, BaseType]
