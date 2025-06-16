from __future__ import annotations
from dataclasses import dataclass

try:
    from .ast_base import ASTNode
except ImportError:
    from ast_base import ASTNode  # type: ignore


@dataclass
class Int(ASTNode):
    value: str


@dataclass
class Float(ASTNode):
    value: str


@dataclass
class String(ASTNode):
    value: str


@dataclass
class True_(ASTNode):
    pass


@dataclass
class False_(ASTNode):
    pass


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
class IdentList(ListOp):
    pass


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
class Negative(UnaryOp):
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
class Powerset(UnaryOp):
    pass


@dataclass
class NonemptyPowerset(UnaryOp):
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
class Type_(ASTNode):
    type_: ASTNode


@dataclass
class Slice(ASTNode):
    items: list[ASTNode]


@dataclass
class UnionAll(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode


@dataclass
class IntersectionAll(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode


@dataclass
class Forall(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode


@dataclass
class Exists(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode


@dataclass
class LambdaDef(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode


@dataclass
class StructAccess(ASTNode):
    struct: ASTNode
    field_name: Identifier


@dataclass
class FunctionCall(ASTNode):
    function_name: ASTNode
    args: list[ASTNode]


@dataclass
class TypedName(ASTNode):
    name: Identifier
    type_: Type_ | None_


@dataclass
class Indexing(ASTNode):
    target: ASTNode
    index: ASTNode | Slice | None_


@dataclass
class Assignment(ASTNode):
    target: ASTNode
    value: ASTNode


@dataclass
class Return(ASTNode):
    value: ASTNode | None_


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
    else_body: Elif | Else | None_


@dataclass
class Elif(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Elif | Else | None_


@dataclass
class For(ASTNode):
    iterable_names: IdentList
    iterable: ASTNode
    body: ASTNode | Statements


@dataclass
class While(ASTNode):
    condition: ASTNode
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
    args: list[TypedName]
    body: ASTNode | Statements
    return_type: Type_


@dataclass
class ImportAll(ASTNode):
    pass


@dataclass
class Import(ASTNode):
    module_identifier: list[Identifier]
    import_objects: IdentList | None_ | ImportAll


@dataclass
class SeqLike(ASTNode):
    items: list[ASTNode]


@dataclass
class MapLike(ASTNode):
    items: list[Maplet]


@dataclass
class SequenceEnumeration(SeqLike):
    pass


@dataclass
class SetEnumeration(SeqLike):
    pass


@dataclass
class BagEnumeration(SeqLike):
    pass


@dataclass
class RelationEnumeration(MapLike):
    pass


@dataclass
class SeqLikeComprehension(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode


@dataclass
class MapLikeComprehension(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode


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
class RelationComprehension(MapLikeComprehension):
    pass


@dataclass
class Start(ASTNode):
    body: Statements | None_


PrimitiveLiteral = Int | Float | String | True_ | False_ | None_
ComplexLiteral = SequenceEnumeration | SetEnumeration | BagEnumeration | RelationEnumeration
ComplexComprehension = SequenceComprehension | SetComprehension | BagComprehension | RelationComprehension
PrimaryStmt = StructAccess | FunctionCall | Indexing | PrimitiveLiteral | ComplexLiteral | ComplexComprehension | Identifier
EquivalenceStmt = BinaryOp | UnaryOp | ListOp | PrimaryStmt
Expr = EquivalenceStmt | LambdaDef
Atom = Identifier | Expr
ControlFlowStmt = Return | Break | Continue | Pass
SimpleStmt = Expr | Assignment | ControlFlowStmt | Import
CompoundStmt = If | For | StructDef | EnumDef | FunctionDef
Predicate = Forall | Exists | ASTNode
