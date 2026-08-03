from dataclasses import dataclass, field
from enum import Enum, auto

from src.mod.data.types.base import BaseType


class IdentifierContext(Enum):
    VARIABLE = auto()
    RECORD = auto()
    RECORD_FIELD = auto()
    PROCEDURE = auto()
    PROCEDURE_PARAMETER = auto()
    QUANTIFICATION_VARIABLE = auto()
    LOOP_VARIABLE = auto()
    LAMBDA_VARIABLE = auto()
    MODULE_IMPORT = auto()
    ENUM = auto()
    ENUM_ITEM = auto()
    TYPE_NAME = auto()
    BUILTIN_TYPE = auto()


@dataclass
class SymbolTableIdentifierEntry:
    id_: int
    scope: int
    name: str
    declared_type: BaseType | None

    """How was this symbol table entry (identifier) declared? Ex. as a variable or as the name of a procedure?"""
    context: IdentifierContext

    def __hash__(self) -> int:
        return hash((self.id_, self.scope))


class ScopeContext(Enum):
    BASE = auto()
    PROCEDURE = auto()
    QUANTIFICATION = auto()
    CONDITIONAL = auto()
    LOOP = auto()
    LAMBDA = auto()
    RECORD = auto()
    NAMESPACE = auto()


@dataclass
class ScopeTableEntry:
    id_: int
    parent: int | None

    """Symbols in this scope"""
    declared_symbols: set[int]

    """Why was a new scope created? Ex. for a procedure body or a for loop body"""
    context: ScopeContext
