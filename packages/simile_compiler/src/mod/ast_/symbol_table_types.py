from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar, Generic, TypeGuard, Literal, ClassVar

from src.mod.ast_.ast_node_operators import (
    CollectionOperator,
    RelationOperator,
    BinaryOperator,
    UnaryOperator,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.mod.ast_.ast_node_base import ASTNode


class SimileTypeError(Exception):
    """Custom exception for Simile type errors."""

    def __init__(self, message: str, node: ASTNode) -> None:
        message = f"Error {node.get_location()} (at node {node}): {message}"

        super().__init__(message)
        self.node = node


class SubstituteSimileTypeAddon:

    def substitute_eq(self, other: SimileType, mapping: dict[str, SimileType] | None = None) -> bool:
        if mapping is None:
            mapping = {}
        return self._substitute_eq(other, mapping)

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        raise NotImplementedError


class BaseSimileType(Enum):
    """Primitive/Atomic Simile types.

    Although these types may be broken down in terms of set theory,
    code generation targets often have extremely efficient pre-existing implementations.
    """

    PosInt = auto()
    Nat = auto()
    Int = auto()
    Float = auto()
    String = auto()
    Bool = auto()
    None_ = auto()
    """Intended for statements without a type, not expressions. For example, a while loop node doesn't have a type."""

    # For unknown types. Right now, built-in generic polymorphic functions will be of this type,
    # since the symbol table is not smart enough to look up the type of called functions.
    # This is just a hack to get "dom" and "ran" to work for now.
    # Any = auto()

    def __repr__(self) -> str:
        return f"SimileType.{self.name}"

    def is_numeric(self) -> bool:
        return self in {BaseSimileType.Int, BaseSimileType.Float, BaseSimileType.PosInt, BaseSimileType.Nat}

    def substitute_eq(self, other: SimileType, mapping: dict[str, SimileType] | None = None) -> bool:
        return self == other


L = TypeVar("L", bound="SimileType")
R = TypeVar("R", bound="SimileType")
T = TypeVar("T", bound="SimileType")


@dataclass(frozen=True)
class GenericType(SubstituteSimileTypeAddon):
    """Generic types are used primarily for resolving generic procedures/functions into a specific type based on context.

    IDs are only locally valid (i.e., introduced by a procedure argument and used by a procedure's return value).
    """

    id_: str

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        if self.id_ not in mapping:
            mapping[self.id_] = other
        return mapping[self.id_] == other


@dataclass(frozen=True)
class PairType(Generic[L, R], SubstituteSimileTypeAddon):
    """Maplet type"""

    left: L
    right: R

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        if not isinstance(other, PairType):
            return False
        return self.left.substitute_eq(other.left, mapping) and self.right.substitute_eq(other.right, mapping)


@dataclass(frozen=True)
class RelationSubTypeMask:
    total: bool
    total_on_range: bool
    one_to_many: bool
    many_to_one: bool

    _subtype_table: ClassVar[dict[tuple[bool, bool, bool, bool], RelationOperator | None]] = {
        (False, False, False, False): RelationOperator.RELATION,
        (False, False, False, True): None,
        (False, False, True, False): RelationOperator.PARTIAL_FUNCTION,
        (False, False, True, True): RelationOperator.PARTIAL_INJECTION,
        (False, True, False, False): RelationOperator.SURJECTIVE_RELATION,
        (False, True, False, True): None,
        (False, True, True, False): RelationOperator.PARTIAL_SURJECTION,
        (False, True, True, True): None,
        (True, False, False, False): RelationOperator.TOTAL_RELATION,
        (True, False, False, True): None,
        (True, False, True, False): RelationOperator.TOTAL_FUNCTION,
        (True, False, True, True): RelationOperator.TOTAL_INJECTION,
        (True, True, False, False): RelationOperator.TOTAL_SURJECTIVE_RELATION,
        (True, True, False, True): None,
        (True, True, True, False): RelationOperator.TOTAL_SURJECTION,
        (True, True, True, True): RelationOperator.BIJECTION,
    }

    @property
    def one_to_one(self) -> bool:
        return self.one_to_many and self.many_to_one

    @staticmethod
    def bag_type() -> RelationSubTypeMask:
        return RelationSubTypeMask(False, False, True, True)

    def get_properties(self) -> dict[str, bool]:
        return {
            "total": self.total,
            "total_on_range": self.total_on_range,
            "one_to_many": self.one_to_many,
            "many_to_one": self.many_to_one,
        }

    def to_relation_operator(self) -> RelationOperator:
        return self._subtype_table.get(tuple(self.get_properties().values()), RelationOperator.RELATION)  # type: ignore

    @classmethod
    def from_relation_operator(cls, relation_operator: RelationOperator | None) -> RelationSubTypeMask:
        if relation_operator is None:
            return RelationSubTypeMask(False, False, False, False)
        inv_subtype_table = {v: k for k, v in cls._subtype_table.items() if v is not None}
        return RelationSubTypeMask(*inv_subtype_table[relation_operator])

    def get_resulting_operator_set_or_unary(self, combining_operator: BinaryOperator | UnaryOperator) -> RelationSubTypeMask:
        new_properties = self.get_properties()
        match combining_operator:
            case BinaryOperator.DOMAIN_RESTRICTION | BinaryOperator.DOMAIN_SUBTRACTION:
                new_properties["total"] = False
                return RelationSubTypeMask(**new_properties)
            case BinaryOperator.RANGE_RESTRICTION | BinaryOperator.RANGE_SUBTRACTION:
                new_properties["total_on_range"] = False
                return RelationSubTypeMask(**new_properties)
            case UnaryOperator.INVERSE:
                new_properties["one_to_many"], new_properties["many_to_one"] = self.many_to_one, self.one_to_many
                new_properties["total"], new_properties["total_on_range"] = self.total_on_range, self.total
                return RelationSubTypeMask(**new_properties)
            case _:
                return self

    def get_resulting_operator_bin_relation(self, other: RelationSubTypeMask, combining_operator: BinaryOperator) -> RelationSubTypeMask:
        match combining_operator:
            case BinaryOperator.RELATION_OVERRIDING:
                new_properties = self.get_properties()
                new_properties["one_to_many"] = self.one_to_many and other.one_to_many
                new_properties["many_to_one"] = self.many_to_one and other.many_to_one
                new_properties["total"] = other.total
                new_properties["total_on_range"] = other.total_on_range
                return RelationSubTypeMask(**new_properties)

            case BinaryOperator.COMPOSITION:
                new_properties = self.get_properties()
                new_properties["one_to_many"] = self.one_to_many and other.one_to_many
                new_properties["many_to_one"] = self.many_to_one and other.many_to_one
                new_properties["total"] = self.total
                new_properties["total_on_range"] = self.total_on_range and other.total_on_range
                return RelationSubTypeMask(**new_properties)

            case _:
                raise ValueError(
                    f"Cannot combine relation operators with {combining_operator.name} operator (types to be combined were: {self} ({self.to_relation_operator().name}), {other} ({other.to_relation_operator().name}))"
                )


@dataclass(frozen=True)
class SetType(Generic[T], SubstituteSimileTypeAddon):
    """Type to represent sets and set-dependent types: bags, relations, sequences, etc."""

    element_type: T
    relation_subtype: RelationSubTypeMask | None = None

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

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        if not isinstance(other, SetType):
            return False
        return self.element_type.substitute_eq(other.element_type, mapping) and self.relation_subtype == other.relation_subtype


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
class StructTypeDef(SubstituteSimileTypeAddon):
    # Internally a (many-to-one) (total on defined fields) function
    fields: dict[str, SimileType]

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        if not isinstance(other, StructTypeDef):
            return False
        return all(f.substitute_eq(o, mapping) for f, o in zip(self.fields.values(), other.fields.values()))


@dataclass(frozen=True)
class EnumTypeDef(SetType[Literal[BaseSimileType.String]]):
    # Internally a set of identifiers
    element_type: Literal[BaseSimileType.String] = BaseSimileType.String
    members: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ProcedureTypeDef(SubstituteSimileTypeAddon):
    arg_types: dict[str, SimileType]
    return_type: SimileType

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        if not isinstance(other, ProcedureTypeDef):
            return False
        return all(f.substitute_eq(o, mapping) for f, o in zip(self.arg_types.values(), other.arg_types.values())) and self.return_type.substitute_eq(other.return_type, mapping)


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
        if isinstance(t, GenericType):
            continue

        if isinstance(t, TypeUnion):
            types_set.update(t.types)
        else:
            types_set.add(t)
    if len(types_set) == 1:
        return types_set.pop()
    return TypeUnion(types=types_set)


@dataclass(frozen=True)
class TypeUnion(SubstituteSimileTypeAddon):
    """OR-selection of types. This type should only be exposed internally, for narrowing purposes"""

    types: set[SimileType]

    def _substitute_eq(self, other: SimileType, mapping: dict[str, SimileType]) -> bool:
        if not isinstance(other, TypeUnion):
            return False
        return all(f.substitute_eq(o, mapping) for f, o in zip(self.types, other.types))


@dataclass(frozen=True)
class ModuleImports(SubstituteSimileTypeAddon):
    # import these objects into the module namespace
    import_objects: dict[str, SimileType]


@dataclass(frozen=True)
class DeferToSymbolTable(SubstituteSimileTypeAddon):
    """Types dependent on this will not be resolved until the analysis phase"""

    lookup_type: str
    """Identifier to look up in table"""


SimileType = BaseSimileType | PairType | StructTypeDef | EnumTypeDef | ProcedureTypeDef | TypeUnion | ModuleImports | DeferToSymbolTable | SetType | GenericType  # | InstanceOfDef
