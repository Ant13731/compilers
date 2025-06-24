from __future__ import annotations
from dataclasses import dataclass, field, Field
from typing import Callable, ClassVar, Any

from soupsieve import match

from src.mod.ast_.ast_node_base import ASTNode, Identifier
from src.mod.ast_.ast_node_operators import (
    BinaryOperator,
    RelationOperator,
    UnaryOperator,
    ListOperator,
    BoolQuantifierOperator,
    QuantifierOperator,
    ControlFlowOperator,
    CollectionOperator,
    Operators,
)
from src.mod.ast_.type_analysis_types import (
    SimileType,
    BaseSimileType,
    PairType,
    SetType,
    StructTypeDef,
    EnumTypeDef,
    ProcedureTypeDef,
    type_union,
    TypeUnion,
    SimileTypeError,
    DeferToSymbolTable,
)


# TODO generate constructors for the typed dataclasses as a sort of shorthand, especially useful for matching/TRS rule creation
@dataclass
class Int(ASTNode):
    value: str

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.Int


@dataclass
class Float(ASTNode):
    value: str

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.Float


@dataclass
class String(ASTNode):
    value: str

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.String


@dataclass
class True_(ASTNode):
    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.Bool


@dataclass
class False_(ASTNode):
    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.Bool


@dataclass
class None_(ASTNode):
    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


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

    @property
    def get_type(self) -> SimileType:
        return SetType(
            element_type=PairType(
                left=BaseSimileType.Int,
                right=type_union(*map(lambda x: x.get_type, self.items)),
            ),
        )


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    right: ASTNode
    op_type: BinaryOperator

    @classmethod
    def construct_with_op(cls, op_type: BinaryOperator) -> Callable[[ASTNode, ASTNode], BinaryOp]:
        return lambda left, right: cls(left=left, right=right, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        l_type = self.left.get_type
        r_type = self.right.get_type
        match self.op_type:
            case BinaryOperator.IMPLIES | BinaryOperator.REV_IMPLIES | BinaryOperator.EQUIVALENT | BinaryOperator.NOT_EQUIVALENT:
                if not (l_type == BaseSimileType.Bool and r_type == BaseSimileType.Bool):
                    raise SimileTypeError(f"Invalid types for logical binary operation: {l_type}, {r_type}")
                return BaseSimileType.Bool
            case BinaryOperator.ADD | BinaryOperator.SUBTRACT:
                union_type = type_union(l_type, r_type)
                match union_type:
                    case BaseSimileType.Int:
                        return BaseSimileType.Int
                    case BaseSimileType.String:
                        return BaseSimileType.String
                    case _ if union_type == SetType:
                        assert isinstance(l_type, SetType) and isinstance(
                            r_type, SetType
                        ), "Both sides must be collections for this operation (checked earlier in BinaryOp.get_type)"

                        if self.op_type == BinaryOperator.ADD:
                            if SetType.is_sequence(l_type) and SetType.is_sequence(r_type):
                                return SetType(
                                    element_type=PairType(
                                        BaseSimileType.Int,
                                        type_union(l_type.element_type.right, r_type.element_type.right),
                                    ),
                                )
                        if SetType.is_bag(l_type) and SetType.is_bag(r_type):
                            return SetType(
                                element_type=PairType(
                                    type_union(l_type.element_type.left, r_type.element_type.left),
                                    BaseSimileType.Int,
                                ),
                            )
                    case _ if union_type == TypeUnion({BaseSimileType.Int, BaseSimileType.Float}):
                        return BaseSimileType.Float

                raise SimileTypeError(f"Invalid types for binary operation {self.op_type.name}: {l_type}, {r_type}")
            case BinaryOperator.MULTIPLY | BinaryOperator.DIVIDE | BinaryOperator.EXPONENT:
                union_type = type_union(l_type, r_type)
                match union_type:
                    case BaseSimileType.Int:
                        return BaseSimileType.Int
                    case _ if union_type == TypeUnion({BaseSimileType.Int, BaseSimileType.Float}):
                        return BaseSimileType.Float
                raise SimileTypeError(f"Invalid types for arithmetic binary operation: {l_type}, {r_type}")
            case BinaryOperator.MODULO:
                if type_union(l_type, r_type) == BaseSimileType.Int:
                    return BaseSimileType.Int
                raise SimileTypeError(f"Invalid types for modulo operation: {l_type}, {r_type}")
            case BinaryOperator.LESS_THAN | BinaryOperator.LESS_THAN_OR_EQUAL | BinaryOperator.GREATER_THAN | BinaryOperator.GREATER_THAN_OR_EQUAL:
                union_type_ = {l_type, r_type}
                if union_type_.issubset({BaseSimileType.Int, BaseSimileType.Float}):
                    return BaseSimileType.Bool
            case BinaryOperator.EQUAL | BinaryOperator.NOT_EQUAL | BinaryOperator.IS | BinaryOperator.IS_NOT:
                return BaseSimileType.Bool
            case BinaryOperator.IN | BinaryOperator.NOT_IN:
                if isinstance(r_type, SetType):
                    return BaseSimileType.Bool
                raise SimileTypeError(f"Invalid types for IN operation: {l_type}, {r_type}")
            case BinaryOperator.UNION | BinaryOperator.INTERSECTION | BinaryOperator.DIFFERENCE:
                if not isinstance(l_type, SetType):
                    raise SimileTypeError(f"Invalid types for set operation (left operand is not a set): {l_type}, {r_type}")
                if not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for set operation (right operand is not a set): {l_type}, {r_type}")
                if SetType.is_relation(l_type):
                    raise SimileTypeError(f"Invalid types for set operation (left operand is a relation): {l_type}, {r_type}")
                if SetType.is_sequence(l_type):
                    raise SimileTypeError(f"Invalid types for set operation (left operand is a sequence): {l_type}, {r_type}")
                if SetType.is_relation(r_type):
                    raise SimileTypeError(f"Invalid types for set operation (right operand is a relation): {l_type}, {r_type}")
                if SetType.is_sequence(r_type):
                    raise SimileTypeError(f"Invalid types for set operation (right operand is a sequence): {l_type}, {r_type}")

                if SetType.is_bag(l_type):
                    if SetType.is_bag(r_type):
                        return SetType(
                            element_type=PairType(
                                type_union(l_type.element_type.left, r_type.element_type.left),
                                BaseSimileType.Int,
                            ),
                        )
                    return SetType(
                        element_type=PairType(
                            type_union(l_type.element_type.left, r_type.element_type),
                            BaseSimileType.Int,
                        ),
                    )
                if SetType.is_bag(r_type):
                    return SetType(
                        element_type=PairType(
                            type_union(l_type.element_type, r_type.element_type.left),
                            BaseSimileType.Int,
                        ),
                    )
                return SetType(
                    element_type=type_union(l_type.element_type, r_type.element_type),
                )

            case (
                BinaryOperator.SUBSET
                | BinaryOperator.SUBSET_EQ
                | BinaryOperator.SUPERSET
                | BinaryOperator.SUPERSET_EQ
                | BinaryOperator.NOT_SUBSET
                | BinaryOperator.NOT_SUBSET_EQ
                | BinaryOperator.NOT_SUPERSET
                | BinaryOperator.NOT_SUPERSET_EQ
            ):
                if not isinstance(l_type, SetType):
                    raise SimileTypeError(f"Invalid types for set operation (left operand is not a set): {l_type}, {r_type}")
                if not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for set operation (right operand is not a set): {l_type}, {r_type}")

                if SetType.is_relation(l_type) or SetType.is_sequence(l_type):
                    raise SimileTypeError(f"Invalid types for subset/superset operation (left operand is a relation or sequence): {l_type}, {r_type}")
                if SetType.is_relation(r_type) or SetType.is_sequence(r_type):
                    raise SimileTypeError(f"Invalid types for subset/superset operation (right operand is a relation or sequence): {l_type}, {r_type}")
                return BaseSimileType.Bool
            case BinaryOperator.MAPLET:
                return PairType(l_type, r_type)
            case BinaryOperator.CARTESIAN_PRODUCT:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for cartesian product operation: {l_type}, {r_type}")

                if not SetType.is_set(l_type) or not SetType.is_set(r_type):
                    raise SimileTypeError(f"Invalid types for cartesian product operation (both operands must be sets): {l_type}, {r_type}")

                return SetType(element_type=PairType(l_type.element_type, r_type.element_type))
            case BinaryOperator.UPTO:
                if not isinstance(l_type, Int) or not isinstance(r_type, Int):
                    raise SimileTypeError(f"Invalid types for upto operation (must be ints): {l_type}, {r_type}")
                return SetType(element_type=BaseSimileType.Int)
            case BinaryOperator.RELATION_OVERRIDING:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for relation operation: {l_type}, {r_type}")

                return SetType(
                    element_type=type_union(l_type.element_type, r_type.element_type),
                )
            case BinaryOperator.COMPOSITION:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for composition operation: {l_type}, {r_type}")
                if not SetType.is_relation(l_type) or not SetType.is_relation(r_type):
                    raise SimileTypeError(f"Invalid collection type for composition operation: {l_type.element_type}, {r_type.element_type}")

                if l_type.element_type.right != r_type.element_type.left:
                    raise SimileTypeError(
                        f"Invalid types for composition operation (right side of left pair does not match with left side of right pair): {l_type.element_type}, {r_type.element_type}"
                    )
                return SetType(
                    element_type=PairType(l_type.element_type.left, r_type.element_type.right),
                )
            case BinaryOperator.DOMAIN_SUBTRACTION | BinaryOperator.DOMAIN_RESTRICTION:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for domain operation: {l_type}, {r_type}")
                if not SetType.is_relation(r_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (right operand must be a relation): {r_type.element_type}")
                if not SetType.is_set(l_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (left operand must be a relation or set): {l_type.element_type}")
                return r_type
            case BinaryOperator.RANGE_SUBTRACTION | BinaryOperator.RANGE_RESTRICTION:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for domain operation: {l_type}, {r_type}")
                if not SetType.is_relation(l_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (left operand must be a relation): {l_type.element_type}")
                if not SetType.is_set(r_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (right operand must be a relation or set): {r_type.element_type}")
                return l_type
        raise SimileTypeError(f"Unknown type for binary operator: {self.op_type.name}, {l_type}, {r_type}")


@dataclass
class RelationOp(ASTNode):
    left: ASTNode
    right: ASTNode
    op_type: RelationOperator

    @classmethod
    def construct_with_op(cls, op_type: RelationOperator) -> Callable[[ASTNode, ASTNode], RelationOp]:
        return lambda left, right: cls(left=left, right=right, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        l_type = self.left.get_type
        r_type = self.right.get_type
        if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
            raise SimileTypeError(f"Invalid types for relation operation: {l_type}, {r_type}")
        # Even if the left/right side of the relation is a set or relation, we just make a new pairtype
        return SetType(element_type=PairType(l_type.element_type, r_type.element_type))


@dataclass
class UnaryOp(ASTNode):
    value: ASTNode
    op_type: UnaryOperator

    @classmethod
    def construct_with_op(cls, op_type: UnaryOperator) -> Callable[[ASTNode], UnaryOp]:
        return lambda value: cls(value=value, op_type=op_type)

    @property
    def bound(self) -> set[Identifier]:
        return self.value.bound

    @property
    def free(self) -> set[Identifier]:
        return self.value.free

    def well_formed(self) -> bool:
        return self.value.well_formed()

    @property
    def get_type(self) -> SimileType:
        match self.op_type:
            case UnaryOperator.NOT:
                if self.value.get_type != BaseSimileType.Bool:
                    raise SimileTypeError(f"Invalid type for NOT operation: {self.value.get_type}")
                return BaseSimileType.Bool
            case UnaryOperator.NEGATIVE:
                if self.value.get_type not in {BaseSimileType.Int, BaseSimileType.Float}:
                    raise SimileTypeError(f"Invalid type for negation: {self.value.get_type}")
                return self.value.get_type
            case UnaryOperator.INVERSE:
                if not isinstance(self.value.get_type, SetType):
                    raise SimileTypeError(f"Invalid type for inverse operation: {self.value.get_type}")
                if not SetType.is_relation(self.value.get_type):
                    raise SimileTypeError(f"Invalid collection type for inverse operation: {self.value.get_type.element_type}")
                return self.value.get_type
            case UnaryOperator.POWERSET | UnaryOperator.NONEMPTY_POWERSET:
                if not isinstance(self.value.get_type, SetType):
                    raise SimileTypeError(f"Invalid type for powerset operation: {self.value.get_type}")
                return SetType(element_type=self.value.get_type)


@dataclass
class ListOp(ASTNode):
    items: list[ASTNode]
    op_type: ListOperator

    @classmethod
    def construct_with_op(cls, op_type: ListOperator) -> Callable[[list[ASTNode]], ListOp]:
        return lambda items: cls(items=items, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        match self.op_type:
            case ListOperator.AND | ListOperator.OR:
                if not all(item.get_type == BaseSimileType.Bool for item in self.items):
                    raise SimileTypeError(f"Invalid types for logical list operation: {[item.get_type for item in self.items]}")
                return BaseSimileType.Bool


@dataclass
class BoolQuantifier(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    op_type: BoolQuantifierOperator

    @classmethod
    def construct_with_op(cls, op_type: BoolQuantifierOperator) -> Callable[[IdentList, ASTNode], BoolQuantifier]:
        return lambda bound_identifiers, predicate: cls(bound_identifiers=bound_identifiers, predicate=predicate, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        if not self.predicate.get_type == BaseSimileType.Bool:
            raise SimileTypeError(f"Invalid type for boolean quantifier predicate: {self.predicate.get_type}")
        return BaseSimileType.Bool


@dataclass
class Quantifier(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode
    op_type: QuantifierOperator

    @classmethod
    def construct_with_op(cls, op_type: QuantifierOperator) -> Callable[[IdentList, ASTNode, ASTNode], Quantifier]:
        return lambda bound_identifiers, predicate, expression: cls(bound_identifiers=bound_identifiers, predicate=predicate, expression=expression, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        if not self.predicate.get_type == BaseSimileType.Bool:
            raise SimileTypeError(f"Invalid type for boolean quantifier predicate: {self.predicate.get_type}")
        # Quantifier operators must be either union all or intersection all, so result is a set
        return SetType(element_type=self.expression.get_type)


@dataclass
class Enumeration(ASTNode):
    items: list[ASTNode]
    op_type: CollectionOperator

    @classmethod
    def construct_with_op(cls, op_type: CollectionOperator) -> Callable[[list[ASTNode]], Enumeration]:
        return lambda items: cls(items=items, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        element_type = type_union(*(item.get_type for item in self.items))
        return SetType(element_type=element_type)


@dataclass
class Comprehension(ASTNode):
    bound_identifiers: IdentList
    predicate: ASTNode
    expression: ASTNode
    op_type: CollectionOperator

    @classmethod
    def construct_with_op(cls, op_type: CollectionOperator) -> Callable[[IdentList, ASTNode, ASTNode], Comprehension]:
        return lambda bound_identifiers, predicate, expression: cls(bound_identifiers=bound_identifiers, predicate=predicate, expression=expression, op_type=op_type)

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

    @property
    def get_type(self) -> SimileType:
        return SetType(element_type=self.expression.get_type)


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

    @property
    def get_type(self) -> SimileType:
        return self.type_.get_type


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

    @property
    def get_type(self) -> SimileType:
        arg_types = {}
        for arg in self.ident_pattern.items:
            if not isinstance(arg, Identifier):
                raise SimileTypeError(f"Invalid lambda argument name (must be an identifier): {arg}")
            arg_types[arg.name] = arg.get_type

        return ProcedureTypeDef(
            arg_types=arg_types,
            return_type=self.expression.get_type,
        )


@dataclass
class StructAccess(ASTNode):
    struct: ASTNode
    field_name: Identifier

    @property
    def get_type(self) -> SimileType:
        if not isinstance(self.struct.get_type, StructTypeDef):
            raise SimileTypeError(f"Struct access target must be a struct type, got {self.struct.get_type}")
        if self.struct.get_type.fields.get(self.field_name.name) is None:
            raise SimileTypeError(f"Field '{self.field_name.name}' not found in struct type")

        return self.struct.get_type.fields.get(self.field_name.name, BaseSimileType.None_)


@dataclass
class Call(ASTNode):
    target: ASTNode
    args: list[ASTNode]

    @property
    def get_type(self) -> SimileType:

        match self.target.get_type:
            case ProcedureTypeDef(arg_types, return_type):
                if len(self.args) != len(arg_types):
                    raise SimileTypeError(f"Argument count mismatch: expected {len(arg_types)}, got {len(self.args)}")
                for i, arg in enumerate(self.args):
                    if arg.get_type != list(arg_types.values())[i]:
                        raise SimileTypeError(f"Argument type mismatch at position {i}: expected {list(arg_types.values())[i]}, got {arg.get_type}")
                return return_type
            case StructTypeDef(arg_types):
                if len(self.args) != len(arg_types):
                    raise SimileTypeError(f"Argument count mismatch: expected {len(arg_types)}, got {len(self.args)}")
                for i, arg in enumerate(self.args):
                    if arg.get_type != list(arg_types.values())[i]:
                        raise SimileTypeError(f"Argument type mismatch at position {i}: expected {list(arg_types.values())[i]}, got {arg.get_type}")
                return StructTypeDef(arg_types)
            case SetType(_) as set_type:
                if not SetType.is_relation(set_type):
                    raise SimileTypeError(f"Cannot call a non-relation collection type: {self.target.get_type}")

                return set_type.element_type.right
            case any_:
                return any_

        raise SimileTypeError(f"Invalid call target type: {self.target.get_type} (must be a procedure, struct, or relation type)")


@dataclass
class Image(ASTNode):
    target: ASTNode
    index: ASTNode

    @property
    def get_type(self) -> SimileType:
        if not isinstance(self.target.get_type, SetType):
            raise SimileTypeError(f"Indexing target must be a collection type, got {self.target.get_type}")

        if not SetType.is_relation(self.target.get_type):
            raise SimileTypeError(f"Indexing target must be a relation type (not set), got {self.target.get_type}")

        return SetType(element_type=self.target.get_type.element_type.right)


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

    @property
    def get_type(self) -> SimileType:
        expected_type = self.name.get_type

        if isinstance(expected_type, DeferToSymbolTable):
            return expected_type

        if expected_type != self.type_.get_type:
            raise SimileTypeError(f"Type mismatch for typed name: expected {expected_type}, got {self.type_.get_type}")

        return expected_type

        # return DeferToSymbolTable(
        #     self.name.get_type,
        #     self.type_.get_type if self.type_ else None,
        #     lambda expected_type: expected_type if expected_type else BaseSimileType.None_,
        # )


@dataclass
class Assignment(ASTNode):
    target: ASTNode
    value: ASTNode

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Return(ASTNode):
    value: ASTNode | None_

    @property
    def get_type(self) -> SimileType:
        return self.value.get_type


@dataclass
class ControlFlowStmt(ASTNode):
    op_type: ControlFlowOperator

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Statements(ASTNode):
    items: list[ASTNode]
    env: Any = None

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Else(ASTNode):
    body: ASTNode | Statements

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class If(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Elif | Else | None_

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Elif(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Elif | Else | None_

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class For(ASTNode):
    iterable_names: IdentList
    iterable: ASTNode
    body: ASTNode | Statements

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class While(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class StructDef(ASTNode):
    name: Identifier
    items: list[TypedName]

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


# @dataclass
# class EnumDef(ASTNode):
#     name: Identifier
#     items: list[Identifier]

#     @property
#     def get_type(self) -> SimileType:
#         return BaseSimileType.None_


@dataclass
class ProcedureDef(ASTNode):
    name: Identifier
    args: list[TypedName]
    body: ASTNode | Statements
    return_type: Type_

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class ImportAll(ASTNode):
    pass

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Import(ASTNode):
    module_file_path: str
    import_objects: IdentList | None_ | ImportAll

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Start(ASTNode):
    body: Statements | None_

    @property
    def get_type(self) -> SimileType:
        return BaseSimileType.None_


Literal = Int | Float | String | True_ | False_ | None_
Predicate = BoolQuantifier | BinaryOp | UnaryOp | True_ | False_
Primary = StructAccess | Call | Image | Literal | Enumeration | Comprehension | Identifier
Expr = LambdaDef | Quantifier | Predicate | BinaryOp | UnaryOp | ListOp | Primary | Identifier
SimpleStmt = Expr | Assignment | ControlFlowStmt | Import
CompoundStmt = If | For | StructDef | ProcedureDef
