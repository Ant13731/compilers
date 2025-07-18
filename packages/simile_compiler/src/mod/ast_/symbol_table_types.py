from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar, Generic, TypeGuard, Literal

from src.mod.ast_.ast_node_operators import CollectionOperator, RelationOperator


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

    # For unknown types. Right now, built-in generic polymorphic functions will be of this type, since the symbol table is not smart enough to look up the type of called functions.
    # This is just a hack to get "dom" and "ran" to work for now.
    Any = auto()

    def __repr__(self) -> str:
        return f"SimileType.{self.name}"


L = TypeVar("L", bound="SimileType")
R = TypeVar("R", bound="SimileType")
T = TypeVar("T", bound="SimileType")


@dataclass(frozen=True)
class PairType(Generic[L, R]):
    left: L
    right: R


@dataclass(frozen=True)
class SetType(Generic[T]):
    element_type: T
    relation_subtype: RelationOperator | None = None

    @staticmethod
    def is_set(self_: SetType) -> bool:
        return not isinstance(self_.element_type, PairType)

    @staticmethod
    def is_relation(self_: SetType) -> TypeGuard[SetType[PairType]]:
        return isinstance(self_.element_type, PairType)

    @staticmethod
    def is_sequence(self_: SetType) -> TypeGuard[SetType[PairType[Literal[BaseSimileType.Int], SimileType]]]:
        return SetType.is_relation(self_) and self_.element_type.left == BaseSimileType.Int

    @staticmethod
    def is_bag(self_: SetType) -> TypeGuard[SetType[PairType[SimileType, Literal[BaseSimileType.Int]]]]:
        return SetType.is_relation(self_) and self_.element_type.right == BaseSimileType.Int


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


@dataclass(frozen=True)
class StructTypeDef:
    # Internally a (many-to-one) (total on defined fields) function
    fields: dict[str, SimileType]


@dataclass(frozen=True)
class EnumTypeDef:
    # Internally a set of identifiers
    members: set[str]


@dataclass(frozen=True)
class ProcedureTypeDef:
    arg_types: dict[str, SimileType]
    return_type: SimileType


# @dataclass
# class InstanceOfDef:
#     type_name: str
#     instance_type: StructTypeDef | EnumTypeDef | ProcedureTypeDef

#     @classmethod
#     def wrap_def_types(cls, type_name: str, instance_type: SimileType) -> SimileType:
#         if isinstance(instance_type, (StructTypeDef, EnumTypeDef, ProcedureTypeDef)):
#             return cls(type_name=type_name, instance_type=instance_type)
#         return instance_type


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


@dataclass(frozen=True)
class TypeUnion:
    types: set[SimileType]


@dataclass(frozen=True)
class ModuleImports:
    # import these objects into the module namespace
    import_objects: dict[str, SimileType]


@dataclass(frozen=True)
class DeferToSymbolTable:
    lookup_type: str


SimileType = BaseSimileType | PairType | StructTypeDef | EnumTypeDef | ProcedureTypeDef | TypeUnion | ModuleImports | DeferToSymbolTable | SetType  # | InstanceOfDef
