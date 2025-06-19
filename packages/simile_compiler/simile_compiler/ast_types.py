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
    items: list[Identifier | IdentList | BinaryOp]  # This binaryop can only be a maplet

    @property
    def free(self) -> set[Identifier]:
        return set(self.find_all_instances(Identifier))

    def well_formed(self) -> bool:
        identifiers = self.find_all_instances(Identifier)
        for i in range(len(identifiers)):
            for j in range(i + 1, len(identifiers)):
                if identifiers[i] == identifiers[j]:
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
                self.left.free.isdisjoint(self.right.bound),
                self.left.bound.isdisjoint(self.right.free),
                self.left.bound.isdisjoint(self.right.bound),
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
                if not item.free.isdisjoint(other.bound):
                    return False
                if not item.bound.isdisjoint(other.free):
                    return False
                if not item.bound.isdisjoint(other.bound):
                    return False
        return True


@dataclass
class BoolQuantifier(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    op_type: BoolQuantifierType

    @property
    def bound(self) -> set[Identifier]:
        return self.bound_identifiers.free | self.predicate.bound

    @property
    def free(self) -> set[Identifier]:
        return self.predicate.free - self.bound_identifiers.free

    def well_formed(self) -> bool:
        return all(
            [
                self.bound_identifiers.well_formed(),
                self.predicate.well_formed(),
                self.predicate.bound.isdisjoint(self.bound_identifiers.free),
            ]
        )


@dataclass
class Quantifier(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode
    op_type: QuantifierType

    @property
    def bound(self) -> set[Identifier]:
        if self.bound_identifiers.items:
            return self.predicate.bound | self.expression.bound | self.bound_identifiers.free
        return self.predicate.bound | self.expression.bound | self.expression.free

    @property
    def free(self) -> set[Identifier]:
        if self.bound_identifiers.items:
            return (self.predicate.free | self.expression.free) - self.bound_identifiers.free
        return self.predicate.free - self.expression.free

    def well_formed(self) -> bool:
        check_list = [
            self.bound_identifiers.well_formed(),
            self.predicate.well_formed(),
            self.expression.well_formed(),
            self.predicate.bound.isdisjoint(self.expression.bound),
            self.predicate.bound.isdisjoint(self.expression.free),
        ]

        if self.bound_identifiers.items:
            check_list += [
                self.predicate.free.isdisjoint(self.expression.bound),
                self.predicate.bound.isdisjoint(self.bound_identifiers.free),
                self.expression.bound.isdisjoint(self.bound_identifiers.free),
            ]

        return all(check_list)


@dataclass
class ControlFlowStmt(ASTNode):
    op_type: ControlFlowType


@dataclass
class Enumeration(ASTNode):
    items: list[ASTNode]
    op_type: CollectionType

    @property
    def bound(self) -> set[Identifier]:
        return set.union(*(item.bound for item in self.items))

    @property
    def free(self) -> set[Identifier]:
        return set.union(*(item.free for item in self.items))

    def well_formed(self) -> bool:
        if not all(item.well_formed() for item in self.items):
            return False
        for i in range(len(self.items)):
            for j in range(len(self.items)):
                if i == j:
                    continue
                # Is this too restrictive? this would block statements like {{x | x > 0}, {x | x > 0}}
                # which may be perfectly valid if x is only locally bound...
                if not self.items[i].bound.isdisjoint(self.items[j].bound):
                    return False
                if not self.items[i].bound.isdisjoint(self.items[j].free):
                    return False
        return True


@dataclass
class Comprehension(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode
    op_type: CollectionType

    @property
    def bound(self) -> set[Identifier]:
        if self.bound_identifiers.items:
            return self.predicate.bound | self.expression.bound | self.bound_identifiers.free
        return self.predicate.bound | self.expression.bound | self.expression.free

    @property
    def free(self) -> set[Identifier]:
        if self.bound_identifiers.items:
            return (self.predicate.free | self.expression.free) - self.bound_identifiers.free
        return self.predicate.free - self.expression.free

    def well_formed(self) -> bool:
        check_list = [
            self.bound_identifiers.well_formed(),
            self.predicate.well_formed(),
            self.expression.well_formed(),
            self.predicate.bound.isdisjoint(self.expression.bound),
            self.predicate.bound.isdisjoint(self.expression.free),
        ]

        if self.bound_identifiers.items:
            check_list += [
                self.predicate.free.isdisjoint(self.expression.bound),
                self.predicate.bound.isdisjoint(self.bound_identifiers.free),
                self.expression.bound.isdisjoint(self.bound_identifiers.free),
            ]

        return all(check_list)


@dataclass
class Type_(ASTNode):
    type_: ASTNode

    @property
    def free(self) -> set[Identifier]:
        return self.type_.free

    @property
    def bound(self) -> set[Identifier]:
        return self.type_.bound

    def well_formed(self) -> bool:
        return self.type_.well_formed()


@dataclass
class Slice(ASTNode):
    items: list[ASTNode]

    @property
    def free(self) -> set[Identifier]:
        return set.union(*(item.free for item in self.items))

    @property
    def bound(self) -> set[Identifier]:
        return set.union(*(item.bound for item in self.items))

    def well_formed(self) -> bool:
        if not all(item.well_formed() for item in self.items):
            return False
        for i in range(len(self.items)):
            for j in range(len(self.items)):
                if i == j:
                    continue
                if not self.items[i].bound.isdisjoint(self.items[j].bound):
                    return False
                if not self.items[i].bound.isdisjoint(self.items[j].free):
                    return False
        return True


@dataclass
class LambdaDef(ASTNode):
    ident_pattern: IdentList
    predicate: ASTNode
    expression: ASTNode

    @property
    def bound(self) -> set[Identifier]:
        return self.ident_pattern.free | self.predicate.bound | self.expression.bound

    @property
    def free(self) -> set[Identifier]:
        return (self.predicate.free | self.expression.free) - self.ident_pattern.free

    def well_formed(self) -> bool:
        return all(
            [
                self.ident_pattern.well_formed(),
                self.predicate.well_formed(),
                self.expression.well_formed(),
                self.predicate.bound.isdisjoint(self.expression.free),
                self.expression.bound.isdisjoint(self.predicate.free),
                self.predicate.bound.isdisjoint(self.expression.bound),
                self.predicate.bound.isdisjoint(self.ident_pattern.free),
                self.expression.bound.isdisjoint(self.ident_pattern.free),
            ]
        )


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
    name: Identifier | ASTNode
    type_: Type_ | None_

    @property
    def free(self) -> set[Identifier]:
        return self.name.free | self.type_.free

    @property
    def bound(self) -> set[Identifier]:
        return self.name.bound | self.type_.bound

    def well_formed(self) -> bool:
        return self.name.well_formed() and self.type_.well_formed()


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
