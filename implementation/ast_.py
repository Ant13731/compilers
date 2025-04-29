from __future__ import annotations
from dataclasses import dataclass, fields, asdict, field
from typing import TypeVar, Generic, Callable, ClassVar, Any, TypeAlias

from llvmlite import ir


@dataclass
class BaseAST:
    """Base class for all AST nodes."""

    def pprint_types(self, indent=0) -> None:
        """Recursively pretty prints the types of the AST node and its fields."""
        indentation = " " * indent
        print(indentation + f"{self.__class__.__name__}:")
        for field in fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, BaseAST):
                print(indentation + f"  {field.name}:")
                field_value.pprint_types(indent + 2)
            elif isinstance(field_value, list):
                print(indentation + f"  {field.name}: [")
                for item in field_value:
                    if isinstance(item, BaseAST):
                        item.pprint_types(indent + 4)
                    else:
                        print(indentation + f"    {item}")
                print(indentation + "  ]")
            else:
                print(indentation + f"  {field.name}: {type(field_value).__name__}")

    def llvm_codegen(self) -> str:
        """Generates LLVM IR code for the AST node."""
        raise NotImplementedError(f"Code generation not implemented for {self.__class__.__name__}.")


# Below collection of classes vaguely matches the rules of grammar.lark.
# For now, most of these are empty classes used only to identify the type of AST node,
# but eventually the idea is to have rewrite rules act on these classes and implement
# a visitor pattern to generate LLVM IR code for each node.


# PRIMITIVE LITERALS, take in a Token and return an AST node
@dataclass
class Int(BaseAST):
    value: int


@dataclass
class Float(BaseAST):
    value: float


@dataclass
class String(BaseAST):
    value: str


@dataclass
class None_(BaseAST):
    pass


@dataclass
class Bool(BaseAST):
    value: bool


@dataclass
class Identifier(BaseAST):
    value: str


# Operators
@dataclass
class BinOp(BaseAST):
    left: PrimaryStmt | UnaryOp | BinOp
    right: PrimaryStmt | UnaryOp | BinOp


@dataclass
class UnaryOp(BaseAST):
    value: PrimaryStmt | UnaryOp | BinOp


# UnaryOp classes
class Neg(UnaryOp):
    pass


class Pos(UnaryOp):
    pass


class Not(UnaryOp):
    pass


class PowerSet(UnaryOp):
    pass


class Complement(UnaryOp):
    pass


# BinOp classes
class Add(BinOp):
    pass


class Sub(BinOp):
    pass


class Mul(BinOp):
    pass


class Div(BinOp):
    pass


class Mod(BinOp):
    pass


class Rem(BinOp):
    pass


class Lt(BinOp):
    pass


class Le(BinOp):
    pass


class Gt(BinOp):
    pass


class Ge(BinOp):
    pass


class Eq(BinOp):
    pass


class Ne(BinOp):
    pass


class In(BinOp):
    pass


class NotIn(BinOp):
    pass


class Is(BinOp):
    pass


class IsNot(BinOp):
    pass


@dataclass
class And(BaseAST):
    clauses: list[PrimaryStmt | UnaryOp | BinOp]


@dataclass
class Or(BaseAST):
    clauses: list[PrimaryStmt | UnaryOp | BinOp]


class Implies(BinOp):
    pass


class RevImplies(BinOp):
    pass


class Equiv(BinOp):
    pass


class NotEquiv(BinOp):
    pass


class Subset(BinOp):
    pass


class SubsetEq(BinOp):
    pass


class Superset(BinOp):
    pass


class SupersetEq(BinOp):
    pass


class Union(BinOp):
    pass


class Intersect(BinOp):
    pass


class Difference(BinOp):
    pass


class CartesianProduct(BinOp):
    pass


class RelationComposition(BinOp):
    pass


class Power(BinOp):
    pass


@dataclass
class EquivalenceStmt(BaseAST):
    value: BinOp | UnaryOp | PrimaryStmt


# Calling
@dataclass
class StructAccess(BaseAST):
    struct: PrimaryStmt
    field_name: Identifier


@dataclass
class Call(BaseAST):
    function: PrimaryStmt
    arguments: Arguments


@dataclass
class Arguments(BaseAST):
    arguments: list[Expr]


@dataclass
class TypedName(BaseAST):
    name: Identifier
    type: Type_


@dataclass
class Indexing(BaseAST):
    target: PrimaryStmt
    index: Slice | EquivalenceStmt | None_ = field(default_factory=lambda: None_())


@dataclass
class Slice(BaseAST):
    slices: list[EquivalenceStmt | None_]


# Statements
@dataclass
class Expr(BaseAST):
    value: EquivalenceStmt | LambdaDef


@dataclass
class Assignment(BaseAST):
    target: TypedName
    value: Expr


@dataclass
class ArgDef(BaseAST):
    args: list[TypedName]


@dataclass
class Return(BaseAST):
    value: Expr | None_


@dataclass
class LambdaDef(BaseAST):
    args: ArgDef
    body: EquivalenceStmt


@dataclass
class Break(BaseAST):
    pass


@dataclass
class Continue(BaseAST):
    pass


@dataclass
class Pass(BaseAST):
    pass


@dataclass
class Type_(BaseAST):
    type_: Expr


# Compound Statements
@dataclass
class If(BaseAST):
    condition: EquivalenceStmt
    body: SimpleStatement | Statements
    else_body: Elif | Else | None_ = field(default_factory=lambda: None_())


@dataclass
class Elif(BaseAST):
    condition: EquivalenceStmt
    body: SimpleStatement | Statements
    else_body: Elif | Else | None_ = field(default_factory=lambda: None_())


@dataclass
class Else(BaseAST):
    body: SimpleStatement | Statements


@dataclass
class For(BaseAST):
    iterable_names: IterableNames
    loop_over: Expr
    body: SimpleStatement | Statements


@dataclass
class Struct(BaseAST):
    name: Identifier
    fields: list[TypedName]


@dataclass
class Enum(BaseAST):
    name: Identifier
    variants: list[Identifier]


@dataclass
class Func(BaseAST):
    name: Identifier
    args: ArgDef
    return_type: Type_
    body: SimpleStatement | Statements


# Complex Literals (collections, iterables, etc.)
@dataclass
class IterableNames(BaseAST):
    names: list[Identifier]


@dataclass
class Tuple(BaseAST):
    elements: list[Expr]


@dataclass
class List(BaseAST):
    elements: list[Expr]


@dataclass
class Dict(BaseAST):
    elements: list[KeyPair]


@dataclass
class Set(BaseAST):
    elements: list[Expr]


@dataclass
class Bag(BaseAST):
    elements: list[Expr]


@dataclass
class SetComprehension(BaseAST):
    elem: Expr
    generator: Expr


@dataclass
class DictComprehension(BaseAST):
    elem: KeyPair
    generator: Expr


@dataclass
class BagComprehension(BaseAST):
    elem: Expr
    generator: Expr


@dataclass
class ListComprehension(BaseAST):
    elem: Expr
    generator: Expr


@dataclass
class KeyPair(BaseAST):
    key: Expr
    value: Expr


# Top level AST nodes
@dataclass
class Statements(BaseAST):
    statements: list[SimpleStatement | CompoundStatement]


@dataclass
class Start(BaseAST):
    body: Statements | None_


# Type aliases for readability
PrimitiveLiteral = Int | Float | String | None_ | Bool
ComplexLiteral = Tuple | List | Dict | Set | Bag
ComplexComprehensionLiteral = SetComprehension | DictComprehension | BagComprehension | ListComprehension

Atom = PrimitiveLiteral | ComplexLiteral | ComplexComprehensionLiteral | Expr
PrimaryStmt = StructAccess | Call | Indexing | Atom

SimpleStatement = Expr | Assignment | Return | Break | Continue | Pass
CompoundStatement = If | For | Struct | Enum | Func
