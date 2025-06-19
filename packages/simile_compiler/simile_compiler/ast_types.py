from __future__ import annotations
from dataclasses import dataclass

try:
    from .ast_base import (
        ASTNode,
        Identifier,
        BinaryOpType,
        RelationTypes,
        UnaryOpType,
        ListBoolType,
        BoolQuantifierType,
        QuantifierType,
        ControlFlowType,
        CollectionType,
    )
except ImportError:
    from ast_base import (  # type: ignore
        ASTNode,
        Identifier,
        BinaryOpType,
        RelationTypes,
        UnaryOpType,
        ListBoolType,
        BoolQuantifierType,
        QuantifierType,
        ControlFlowType,
        CollectionType,
    )

# TODO generate constructors for the typed dataclasses
# as a sort of shorthand. maybe even make it a class method that returns a partial func
# def constructor(cls, op_type) -> Callable... 
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
class IdentList(ASTNode):
    items: list[Identifier]

    def __post_init__(self):
        self.with_free = set(self.items)

    def well_formed(self) -> bool:
        for i in range(len(self.items)):
            for j in range(i + 1, len(self.items)):
                if i == j:
                    continue
                if self.items[i] == self.items[j]:
                    return False
        return True


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    right: ASTNode
    op_type: BinaryOpType

    @property
    def bound(self) -> set[Identifier]:
        return self.left.bound | self.right.bound

    @property
    def free(self) -> set[Identifier]:
        return self.left.free | self.right.free

    def well_formed(self) -> bool:
        return all(
            [
                self.left.well_formed(),
                self.right.well_formed(),
                self.left.free & self.right.bound == set(),
                self.left.bound & self.right.free == set(),
                self.left.bound & self.right.bound == set(),
            ]
        )


@dataclass
class RelationOp(ASTNode):
    left: ASTNode
    right: ASTNode
    op_type: RelationTypes

    @property
    def bound(self) -> set[Identifier]:
        return self.left.bound | self.right.bound

    @property
    def free(self) -> set[Identifier]:
        return self.left.free | self.right.free

    def well_formed(self) -> bool:
        return all(
            [
                self.left.well_formed(),
                self.right.well_formed(),
                self.left.free & self.right.bound == set(),
                self.left.bound & self.right.free == set(),
                self.left.bound & self.right.bound == set(),
            ]
        )


@dataclass
class UnaryOp(ASTNode):
    value: ASTNode
    op_type: UnaryOpType

    @property
    def bound(self) -> set[Identifier]:
        return self.value.bound

    @property
    def free(self) -> set[Identifier]:
        return self.value.free

    def well_formed(self) -> bool:
        return self.value.well_formed()


@dataclass
class ListOp(ASTNode):
    items: list[ASTNode]
    op_type: ListBoolType

    @property
    def bound(self) -> set[Identifier]:
        return set.union(*(item.bound for item in self.items))

    @property
    def free(self) -> set[Identifier]:
        return set.union(*(item.free for item in self.items))

    def well_formed(self) -> bool:
        if not all(item.well_formed() for item in self.items):
            return False
        for item in self.items:
            for other in self.items:
                if item == other:
                    continue
                if item.free & other.bound != set():
                    return False
                if item.bound & other.free != set():
                    return False
                if item.bound & other.bound != set():
                    return False
        return True


@dataclass
class BoolQuantifier(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    op_type: BoolQuantifierType


@dataclass
class Quantifier(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode
    op_type: QuantifierType


@dataclass
class ControlFlowStmt(ASTNode):
    op_type: ControlFlowType


@dataclass
class Enumeration(ASTNode):
    items: list[ASTNode]
    op_type: CollectionType


@dataclass
class Comprehension(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode
    op_type: CollectionType


@dataclass
class Type_(ASTNode):
    type_: ASTNode


@dataclass
class Slice(ASTNode):
    items: list[ASTNode]


@dataclass
class LambdaDef(ASTNode):
    ident_pattern: list[ASTNode]
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
class Start(ASTNode):
    body: Statements | None_


Literal = Int | Float | String | True_ | False_ | None_
Predicate = BoolQuantifier | BinaryOp | UnaryOp | True_ | False_
Primary = StructAccess | FunctionCall | Indexing | Literal | Enumeration | Comprehension | Identifier
Expr = LambdaDef | Quantifier | Predicate | BinaryOp | UnaryOp | ListOp | Primary | Identifier
SimpleStmt = Expr | Assignment | ControlFlowStmt | Import
CompoundStmt = If | For | StructDef | EnumDef | FunctionDef
