from __future__ import annotations
from dataclasses import dataclass

try:
    from .ast_base import ASTNode
except ImportError:
    from ast_base import ASTNode  # type: ignore


@dataclass
class Int(ASTNode):
    value: int


@dataclass
class Float(ASTNode):
    value: float


@dataclass
class String(ASTNode):
    value: str


@dataclass
class Bool(ASTNode):
    value: bool


@dataclass
class None_(ASTNode):
    pass


@dataclass
class Identifier(ASTNode):
    name: str


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    right: ASTNode


@dataclass
class UnaryOp(ASTNode):
    value: ASTNode


@dataclass
class ListOp(ASTNode):
    items: list[ASTNode]


@dataclass
class And(ListOp):
    pass


@dataclass
class Or(ListOp):
    pass


@dataclass
class Not(UnaryOp):
    pass


@dataclass
class Implies(BinaryOp):
    pass


@dataclass
class RevImplies(BinaryOp):
    pass


@dataclass
class Equivalent(BinaryOp):
    pass


@dataclass
class NotEquivalent(BinaryOp):
    pass


@dataclass
class Add(BinaryOp):
    pass


@dataclass
class Subtract(BinaryOp):
    pass


@dataclass
class Multiply(BinaryOp):
    pass


@dataclass
class Divide(BinaryOp):
    pass


@dataclass
class Modulus(BinaryOp):
    pass


@dataclass
class Exponent(BinaryOp):
    pass


@dataclass
class Equal(BinaryOp):
    pass


@dataclass
class NotEqual(BinaryOp):
    pass


@dataclass
class LessThan(BinaryOp):
    pass


@dataclass
class LessThanOrEqual(BinaryOp):
    pass


@dataclass
class GreaterThan(BinaryOp):
    pass


@dataclass
class GreaterThanOrEqual(BinaryOp):
    pass


@dataclass
class Is(BinaryOp):
    pass


@dataclass
class IsNot(BinaryOp):
    pass


@dataclass
class In(BinaryOp):
    pass


@dataclass
class NotIn(BinaryOp):
    pass


@dataclass
class Union(BinaryOp):
    pass


@dataclass
class Intersection(BinaryOp):
    pass


@dataclass
class Difference(BinaryOp):
    pass


@dataclass
class Subset(BinaryOp):
    pass


@dataclass
class SubsetEq(BinaryOp):
    pass


@dataclass
class Superset(BinaryOp):
    pass


@dataclass
class SupersetEq(BinaryOp):
    pass


@dataclass
class NotSubset(BinaryOp):
    pass


@dataclass
class NotSubsetEq(BinaryOp):
    pass


@dataclass
class NotSuperset(BinaryOp):
    pass


@dataclass
class NotSupersetEq(BinaryOp):
    pass


@dataclass
class UnionAll(ListOp):
    pass


@dataclass
class IntersectionAll(ListOp):
    pass


@dataclass
class Powerset(UnaryOp):
    pass


@dataclass
class Maplet(BinaryOp):
    pass


@dataclass
class RelationOverride(BinaryOp):
    pass


@dataclass
class Composition(BinaryOp):
    pass


@dataclass
class CartesianProduct(BinaryOp):
    pass


@dataclass
class Inverse(UnaryOp):
    pass


@dataclass
class DomainSubtraction(BinaryOp):
    pass


@dataclass
class DomainRestriction(BinaryOp):
    pass


@dataclass
class RangeSubtraction(BinaryOp):
    pass


@dataclass
class RangeRestriction(BinaryOp):
    pass


@dataclass
class RelationOp(BinaryOp):
    pass


@dataclass
class TotalRelationOp(BinaryOp):
    pass


@dataclass
class SurjectiveRelationOp(BinaryOp):
    pass


@dataclass
class TotalSurjectiveRelation(BinaryOp):
    pass


@dataclass
class PartialFunction(BinaryOp):
    pass


@dataclass
class TotalFunction(BinaryOp):
    pass


@dataclass
class PartialInjection(BinaryOp):
    pass


@dataclass
class TotalInjection(BinaryOp):
    pass


@dataclass
class PartialSurjection(BinaryOp):
    pass


@dataclass
class TotalSurjection(BinaryOp):
    pass


@dataclass
class Bijection(BinaryOp):
    pass


@dataclass
class UpTo(BinaryOp):
    pass


@dataclass
class Arguments(ASTNode):
    items: list[ASTNode]


@dataclass
class Type_(ASTNode):
    type_: ASTNode


@dataclass
class Slice(ASTNode):
    items: list[ASTNode]


@dataclass
class StructAccess(ASTNode):
    struct: ASTNode
    field_name: Identifier


@dataclass
class FunctionCall(ASTNode):
    function_name: ASTNode
    args: Arguments


@dataclass
class TypedName(ASTNode):
    name: Identifier
    type_: Type_ | None_


@dataclass
class Indexing(ASTNode):
    target: ASTNode
    index: ASTNode | Slice | None_


@dataclass
class ArgDef(ASTNode):
    items: list[TypedName]


@dataclass
class Assignment(ASTNode):
    target: ASTNode
    value: ASTNode


@dataclass
class Return(ASTNode):
    value: ASTNode | None_


@dataclass
class LambdaDef(ASTNode):
    args: ArgDef
    body: ASTNode


@dataclass
class Break(ASTNode):
    pass


@dataclass
class Continue(ASTNode):
    pass


@dataclass
class Pass(ASTNode):
    pass


@dataclass
class Statements(ASTNode):
    items: list[ASTNode]


@dataclass
class Else(ASTNode):
    body: ASTNode | Statements


@dataclass
class If(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Else | None_


@dataclass
class Elif(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Elif | Else | None_


@dataclass
class For(ASTNode):
    iterable_names: list[Identifier]
    iterable: ASTNode
    body: ASTNode | Statements


@dataclass
class StructDef(ASTNode):
    name: Identifier
    items: list[TypedName]


@dataclass
class EnumDef(ASTNode):
    name: Identifier
    items: list[Identifier]


@dataclass
class FunctionDef(ASTNode):
    name: Identifier
    args: ArgDef
    body: ASTNode | Statements
    return_type: Type_ | None_


@dataclass
class KeyPair(ASTNode):
    key: ASTNode
    value: ASTNode


@dataclass
class SeqLike(ASTNode):
    items: list[ASTNode]


@dataclass
class MapLike(ASTNode):
    items: list[KeyPair]


@dataclass
class SequenceLiteral(SeqLike):
    pass


@dataclass
class SetLiteral(SeqLike):
    pass


@dataclass
class BagLiteral(MapLike):
    pass


@dataclass
class MappingLiteral(MapLike):
    pass


@dataclass
class SeqLikeComprehension(ASTNode):
    bound_identifiers: list[Identifier]
    generator: ASTNode
    mapping_expression: ASTNode | None_


@dataclass
class MapLikeComprehension(ASTNode):
    bound_identifiers: list[KeyPair]
    generator: ASTNode
    mapping_expression: ASTNode | None_


@dataclass
class SequenceComprehension(SeqLikeComprehension):
    pass


@dataclass
class SetComprehension(SeqLikeComprehension):
    pass


@dataclass
class BagComprehension(SeqLikeComprehension):
    pass


@dataclass
class MappingComprehension(MapLikeComprehension):
    pass


@dataclass
class Start(ASTNode):
    body: Statements | None_


PrimitiveLiteral = Int | Float | String | Bool | None_
ComplexLiteral = SequenceLiteral | SetLiteral | BagLiteral | MappingLiteral
ComplexComprehension = SequenceComprehension | SetComprehension | BagComprehension | MappingComprehension
PrimaryStmt = StructAccess | FunctionCall | Indexing | PrimitiveLiteral | ComplexLiteral | ComplexComprehension | Identifier
EquivalenceStmt = BinaryOp | UnaryOp | ListOp | PrimaryStmt
Expr = EquivalenceStmt | LambdaDef
Atom = Identifier | Expr
SimpleStmt = Expr | Assignment | Return | Break | Continue | Pass
CompoundStmt = If | For | StructDef | EnumDef | FunctionDef
