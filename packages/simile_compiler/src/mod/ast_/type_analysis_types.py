from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar, Generic, TypeGuard, Literal

from src.mod.ast_.ast_node_operators import CollectionOperator


class SimileTypeError(Exception):
    """Custom exception for Simile type errors."""

    pass


class BaseSimileType(Enum):
    PosInt = auto()
    Nat = auto()
    Int = auto()
    Float = auto()
    String = auto()
    Bool = auto()
    None_ = auto()

    def __repr__(self) -> str:
        return f"SimileType.{self.name}"


L = TypeVar("L", bound="SimileType")
R = TypeVar("R", bound="SimileType")
T = TypeVar("T", bound="SimileType")


@dataclass
class PairType(Generic[L, R]):
    left: L
    right: R


@dataclass
class SetType(Generic[T]):
    element_type: T

    @staticmethod
    def is_set(self: SetType) -> bool:
        return not isinstance(self.element_type, PairType)

    @staticmethod
    def is_relation(self: SetType) -> TypeGuard[SetType[PairType]]:
        return isinstance(self.element_type, PairType)

    @staticmethod
    def is_sequence(self: SetType) -> TypeGuard[SetType[PairType[Literal[BaseSimileType.Int], SimileType]]]:
        return SetType.is_relation(self) and self.element_type.left == BaseSimileType.Int

    @staticmethod
    def is_bag(self: SetType) -> TypeGuard[SetType[PairType[SimileType, Literal[BaseSimileType.Int]]]]:
        return SetType.is_relation(self) and self.element_type.right == BaseSimileType.Int


# TODO:
# - base types off of sets
# - enums are sets of (free) identifiers
#   EnumTypeName: enum := {a, b, c}
# - use set theory for relations, functions, etc.
#   - function call is just imaging with hilberts choice
#       - a nondeterministic choice of the resulting image
# - None should not be an object?

# Try to stick with pairs and sets - set theory
# function is a special kind of relation, so all relation operators
# lookup hilberts choice
# hiberts choice on a set is random but the same: epsilon(Relation(a)) = epsilon(Relation(a)) always, but different runs may be different (nondeterminism)
# use functions in the set theory sense
# call anything with side effects/imperative a procedure, not a function
# At some point we may need to include nondeterminism - resolve nondeterminism as late as possible so a nondeterministic set can be implemented as an array, for example
# enums defined as a set of identifiers ()


# c = 5
# x := {a,b,c}

# can write TYPE S = {a,b,c} for new enum
# or SET S = {a,b,c} for set assignment


@dataclass
class StructTypeDef:
    # Internally a (many-to-one) (total on defined fields) function
    fields: dict[str, SimileType]


@dataclass
class EnumTypeDef:
    # Internally a set of identifiers
    members: set[str]


@dataclass
class ProcedureTypeDef:
    arg_types: dict[str, SimileType]
    return_type: SimileType


def type_union(*types: SimileType) -> SimileType:
    """Create a single type or TypeUnion from multiple SimileTypes."""
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
    types: set[SimileType]


@dataclass
class ModuleImports:
    # import these objects into the module namespace
    import_objects: dict[str, SimileType]


@dataclass
class DeferToSymbolTable:
    lookup_type: str


SimileType = BaseSimileType | PairType | StructTypeDef | EnumTypeDef | ProcedureTypeDef | TypeUnion | ModuleImports | DeferToSymbolTable | SetType  # | TypeOf
