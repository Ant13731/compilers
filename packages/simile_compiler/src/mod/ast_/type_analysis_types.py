from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar

from src.mod.ast_.ast_node_operators import CollectionOperator


class SimileTypeError(Exception):
    """Custom exception for Simile type errors."""

    pass


class BaseSimileType(Enum):
    Int = auto()
    Float = auto()
    String = auto()
    Bool = auto()
    None_ = auto()


@dataclass
class PairType:
    left: SimileType
    right: SimileType


@dataclass
class CollectionType:
    element_type: SimileType
    collection_type: CollectionOperator


@dataclass
class StructTypeDef:
    fields: dict[str, SimileType] = field(default_factory=dict)


@dataclass
class EnumTypeDef:
    members: list[str] = field(default_factory=list)


@dataclass
class CustomType:
    name_of_custom_type: str  # Defer lookup in symbol table


@dataclass
class FunctionTypeDef:
    arg_types: list[SimileType] = field(default_factory=list)
    return_type: SimileType


def type_union(*types: SimileType) -> SimileType:
    """Create a TypeUnion from multiple SimileTypes."""
    types_set = set()
    for t in types:
        if isinstance(t, TypeUnion):
            types_set.update(t.types)
        else:
            types_set.add(t)
    if len(types_set) == 1:
        return types_set.pop()
    return TypeUnion(types=types_set)


@dataclass
class TypeUnion:
    types: set[SimileType] = field(default_factory=set)


@dataclass
class ModuleImports:
    import_objects: dict[str, SimileType] = field(default_factory=dict)


T = TypeVar("T", bound=SimileType)


@dataclass
class DeferToSymbolTable:
    lookup_type: SimileType | str
    expected_type: T | None = None
    operation_on_expected_type: Callable[[T], SimileType] | None = None


SimileType = BaseSimileType | PairType | CollectionType | CustomType | StructTypeDef | EnumTypeDef | FunctionTypeDef | TypeUnion | DeferToSymbolTable
