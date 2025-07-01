from __future__ import annotations
from dataclasses import dataclass, field, Field, fields, is_dataclass
from typing import Callable, ClassVar, Any, Self

from soupsieve import match

from src.mod.ast_.ast_node_base import ASTNode, Identifier
from src.mod.ast_.ast_node_operators import (
    BinaryOperator,
    RelationOperator,
    UnaryOperator,
    ListOperator,
    QuantifierOperator,
    ControlFlowOperator,
    CollectionOperator,
    Operators,
)
from src.mod.ast_.symbol_table_types import (
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
# Generate them from enums directly - ex. print(type_writer(op.name.capitalize() for op in BinaryOperator))
@dataclass
class Int(ASTNode):
    value: str

    def _get_type(self) -> SimileType:
        return BaseSimileType.Int


@dataclass
class Float(ASTNode):
    value: str

    def _get_type(self) -> SimileType:
        return BaseSimileType.Float


@dataclass
class String(ASTNode):
    value: str

    def _get_type(self) -> SimileType:
        return BaseSimileType.String


@dataclass
class True_(ASTNode):
    def _get_type(self) -> SimileType:
        return BaseSimileType.Bool


@dataclass
class False_(ASTNode):
    def _get_type(self) -> SimileType:
        return BaseSimileType.Bool


@dataclass
class None_(ASTNode):
    def _get_type(self) -> SimileType:
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

    def _get_type(self) -> SimileType:
        return SetType(
            element_type=PairType(
                left=BaseSimileType.Int,
                right=type_union(*map(lambda x: x.get_type, self.items)),
            ),
        )


class InheritedEqMixin:
    """Introduces structural equality between super/subclasses that ignore type names.

    Only needs to be mixed in the parent class. Must be used with dataclasses."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        assert is_dataclass(self), "InheritedEqMixin can only be used with dataclasses"

        if len(fields(self)) != len(fields(other)):
            return False

        for f in fields(self):
            if f.name.startswith("_"):
                continue
            self_value = getattr(self, f.name)
            try:
                other_value = getattr(other, f.name)
            except AttributeError:
                return False
            if self_value != other_value:
                return False
        return True


@dataclass(eq=False)
class BinaryOp(InheritedEqMixin, ASTNode):
    left: ASTNode
    right: ASTNode
    op_type: BinaryOperator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            self_value = getattr(self, f.name)
            try:
                other_value = getattr(other, f.name)
            except AttributeError:
                return False
            if self_value != other_value:
                return False
        return True

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

    def _get_type(self) -> SimileType:
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

                if SetType.is_relation(l_type):
                    raise SimileTypeError(f"Invalid types for set operation (left operand is a relation): {l_type}, {r_type}")
                if SetType.is_sequence(l_type):
                    raise SimileTypeError(f"Invalid types for set operation (left operand is a sequence): {l_type}, {r_type}")
                if SetType.is_relation(r_type):
                    raise SimileTypeError(f"Invalid types for set operation (right operand is a relation): {l_type}, {r_type}")
                if SetType.is_sequence(r_type):
                    raise SimileTypeError(f"Invalid types for set operation (right operand is a sequence): {l_type}, {r_type}")

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

                return SetType(
                    element_type=PairType(l_type.element_type, r_type.element_type),
                    relation_subtype=RelationOperator.RELATION,
                )
            case BinaryOperator.UPTO:
                if not isinstance(l_type, Int) or not isinstance(r_type, Int):
                    raise SimileTypeError(f"Invalid types for upto operation (must be ints): {l_type}, {r_type}")
                return SetType(element_type=BaseSimileType.Int)
            case BinaryOperator.RELATION_OVERRIDING:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for relation operation: {l_type}, {r_type}")
                relation_subtype_l = l_type.relation_subtype
                relation_subtype_r = r_type.relation_subtype
                if relation_subtype_l is None:
                    relation_subtype_l = RelationOperator.RELATION
                if relation_subtype_r is None:
                    relation_subtype_r = RelationOperator.RELATION
                relation_subtype = relation_subtype_l.get_resulting_operator(relation_subtype_r, BinaryOperator.RELATION_OVERRIDING)

                return SetType(
                    element_type=type_union(l_type.element_type, r_type.element_type),
                    relation_subtype=relation_subtype,
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
                relation_subtype_l = l_type.relation_subtype
                relation_subtype_r = r_type.relation_subtype
                if relation_subtype_l is None:
                    relation_subtype_l = RelationOperator.RELATION
                if relation_subtype_r is None:
                    relation_subtype_r = RelationOperator.RELATION
                relation_subtype = relation_subtype_l.get_resulting_operator(relation_subtype_r, BinaryOperator.COMPOSITION)

                return SetType(
                    element_type=PairType(l_type.element_type.left, r_type.element_type.right),
                    relation_subtype=relation_subtype,
                )
            case BinaryOperator.DOMAIN_SUBTRACTION | BinaryOperator.DOMAIN_RESTRICTION:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for domain operation: {l_type}, {r_type}")
                if not SetType.is_relation(r_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (right operand must be a relation): {r_type.element_type}")
                if not SetType.is_set(l_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (left operand must be a relation or set): {l_type.element_type}")
                if r_type.relation_subtype is not None:
                    relation_subtype = r_type.relation_subtype.get_resulting_operator_set_or_unary(BinaryOperator.DOMAIN_SUBTRACTION)
                return SetType(
                    element_type=r_type.element_type,
                    relation_subtype=relation_subtype,
                )
            case BinaryOperator.RANGE_SUBTRACTION | BinaryOperator.RANGE_RESTRICTION:
                if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
                    raise SimileTypeError(f"Invalid types for domain operation: {l_type}, {r_type}")
                if not SetType.is_relation(l_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (left operand must be a relation): {l_type.element_type}")
                if not SetType.is_set(r_type):
                    raise SimileTypeError(f"Invalid collection type for domain operation (right operand must be a relation or set): {r_type.element_type}")
                if l_type.relation_subtype is not None:
                    relation_subtype = l_type.relation_subtype.get_resulting_operator_set_or_unary(BinaryOperator.RANGE_SUBTRACTION)
                return SetType(
                    element_type=l_type.element_type,
                    relation_subtype=relation_subtype,
                )
        raise SimileTypeError(f"Unknown type for binary operator: {self.op_type.name}, {l_type}, {r_type}")


@dataclass(eq=False)
class RelationOp(InheritedEqMixin, ASTNode):
    left: ASTNode
    right: ASTNode
    op_type: RelationOperator

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

    def _get_type(self) -> SimileType:
        l_type = self.left.get_type
        r_type = self.right.get_type
        if not isinstance(l_type, SetType) or not isinstance(r_type, SetType):
            raise SimileTypeError(f"Invalid types for relation operation: {l_type}, {r_type}")
        # Even if the left/right side of the relation is a set or relation, we just make a new pairtype
        return SetType(element_type=PairType(l_type.element_type, r_type.element_type))


@dataclass(eq=False)
class UnaryOp(InheritedEqMixin, ASTNode):
    value: ASTNode
    op_type: UnaryOperator

    @property
    def bound(self) -> set[Identifier]:
        return self.value.bound

    @property
    def free(self) -> set[Identifier]:
        return self.value.free

    def well_formed(self) -> bool:
        return self.value.well_formed()

    def _get_type(self) -> SimileType:
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

                relation_subtype = self.value.get_type.relation_subtype
                if relation_subtype is not None:
                    relation_subtype = relation_subtype.get_resulting_operator_set_or_unary(UnaryOperator.INVERSE)
                return SetType(
                    element_type=PairType(
                        self.value.get_type.element_type.right,
                        self.value.get_type.element_type.left,
                    ),
                    relation_subtype=relation_subtype,
                )
            case UnaryOperator.POWERSET | UnaryOperator.NONEMPTY_POWERSET:
                if not isinstance(self.value.get_type, SetType):
                    raise SimileTypeError(f"Invalid type for powerset operation: {self.value.get_type}")
                return SetType(element_type=self.value.get_type)


@dataclass(eq=False)
class ListOp(InheritedEqMixin, ASTNode):
    items: list[ASTNode]
    op_type: ListOperator

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

    def _get_type(self) -> SimileType:
        match self.op_type:
            case ListOperator.AND | ListOperator.OR:
                if not all(item.get_type == BaseSimileType.Bool for item in self.items):
                    raise SimileTypeError(f"Invalid types for logical list operation: {[item.get_type for item in self.items]}")
                return BaseSimileType.Bool

    def separate_candidate_generators_from_predicates(self, bound_quantifier_variables: set[Identifier] | None = None) -> tuple[list[BinaryOp], list[ASTNode]]:
        """Get candidate generators from a list of AND-separated predicates (intended for use in comprehension/quantifier based rewrite rules)"""
        if not self.op_type == ListOperator.AND:
            raise SimileTypeError(f"ListOp.get_candidate_generators() can only be called on AND operations, got {self.op_type.name}")

        candidate_generators: list[BinaryOp] = []
        predicates: list[ASTNode] = []
        for item in self.items:
            match item:
                case BinaryOp(
                    Identifier(_) as x,
                    set_type,
                    BinaryOperator.IN,
                ) if isinstance(
                    set_type.get_type, SetType
                ) and (bound_quantifier_variables is None or x in bound_quantifier_variables):
                    candidate_generators.append(item)
                case _:
                    predicates.append(item)
        return candidate_generators, predicates

    @classmethod
    def flatten_and_join(cls, obj_lst: list[Any], type_: ListOperator) -> Self:
        """Flatten a list of objects and join them with the given ListOp."""
        flattened_objs = []
        for obj in obj_lst:
            if isinstance(obj, ListOp) and obj.op_type == type_:
                flattened_objs += obj.items
            else:
                flattened_objs.append(obj)
        return cls(items=flattened_objs, op_type=type_)


@dataclass(eq=False)
class Quantifier(ASTNode):
    predicate: ListOp  # includes generators
    expression: ASTNode
    op_type: QuantifierOperator
    # demoted_predicate: ListOp | None = None  # guaranteed to NOT include generators - should only be filled in with an AND

    def __post_init__(self) -> None:
        super().__post_init__()
        self._bound_identifiers: IdentList | None = None

        # Every quantifier must have one generator per OR clause
        # All OR clauses will be on the top level
        self._selected_generators: list[BinaryOp] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if len(fields(self)) != len(fields(other)):
            return False

        for f in fields(self):
            if f.name not in map(lambda x: x.name, fields(other)):
                return False

        if self.op_type != other.op_type:
            return False
        if self.all_predicates != other.all_predicates:
            return False
        if self.expression != other.expression:
            return False

        for f in fields(self):
            if f.name.startswith("_"):
                continue
            self_value = getattr(self, f.name)
            try:
                other_value = getattr(other, f.name)
            except AttributeError:
                return False
            if self_value != other_value:
                return False
        return True

    @property
    def all_predicates(self) -> ASTNode:
        """Get all predicates in the quantifier, including demoted predicates."""
        predicates = self.predicate
        # if self.demoted_predicate:
        # predicates = ListOp.flatten_and_join([self.predicate, self.demoted_predicate], ListOperator.AND)
        return predicates

    @property
    def bound(self) -> set[Identifier]:
        if self._bound_identifiers and self._bound_identifiers.items:
            return self.all_predicates.bound | self.expression.bound | self._bound_identifiers.free
        return self.all_predicates.bound | self.expression.bound | self.expression.free

    @property
    def free(self) -> set[Identifier]:
        if self._bound_identifiers and self._bound_identifiers.items:
            return (self.all_predicates.free | self.expression.free) - self._bound_identifiers.free
        return self.all_predicates.free - self.expression.free

    def well_formed(self) -> bool:
        check_list = [
            self.all_predicates.well_formed(),
            self.expression.well_formed(),
            self.all_predicates.bound.isdisjoint(self.expression.bound),
            self.all_predicates.bound.isdisjoint(self.expression.free),
        ]

        if self._bound_identifiers:
            check_list += [self._bound_identifiers.well_formed()]

        if self._bound_identifiers and self._bound_identifiers.items:
            check_list += [
                self.all_predicates.free.isdisjoint(self.expression.bound),
                self.all_predicates.bound.isdisjoint(self._bound_identifiers.free),
                self.expression.bound.isdisjoint(self._bound_identifiers.free),
            ]

        return all(check_list)

    def _get_type(self) -> SimileType:
        if not self.all_predicates.get_type == BaseSimileType.Bool:
            raise SimileTypeError(f"Invalid type for boolean quantifier predicate: {self.all_predicates.get_type}")

        if self.op_type.is_bool_quantifier():
            return BaseSimileType.Bool
        if self.op_type.is_collection_operator():
            return SetType(element_type=self.expression.get_type)
        if self.op_type.is_general_collection_operator():
            if not isinstance(self.expression.get_type, SetType):
                raise SimileTypeError(f"Invalid type for general collection operator: (requires SetType in expression) {self.expression.get_type}")
            return self.expression.get_type
        if self.op_type.is_numerical_quantifier():
            return self.expression.get_type

        raise SimileTypeError(f"Invalid quantifier operator type. Expected one of {list(QuantifierOperator)} but got {self.op_type}")


@dataclass(eq=False)
class Enumeration(InheritedEqMixin, ASTNode):
    items: list[ASTNode]
    op_type: CollectionOperator

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

    def _get_type(self) -> SimileType:
        element_type = type_union(*(item.get_type for item in self.items))
        return SetType(element_type=element_type)


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

    def _get_type(self) -> SimileType:
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

    def _get_type(self) -> SimileType:
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

    def _get_type(self) -> SimileType:
        if not isinstance(self.struct.get_type, StructTypeDef):
            raise SimileTypeError(f"Struct access target must be a struct type, got {self.struct.get_type}")
        if self.struct.get_type.fields.get(self.field_name.name) is None:
            raise SimileTypeError(f"Field '{self.field_name.name}' not found in struct type")

        return self.struct.get_type.fields.get(self.field_name.name, BaseSimileType.None_)


@dataclass
class Call(ASTNode):
    target: ASTNode
    args: list[ASTNode]

    def _get_type(self) -> SimileType:

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

        raise SimileTypeError(f"Invalid call target type: {self.target.get_type} (must be a procedure, struct, or relation type)")


@dataclass
class Image(ASTNode):
    target: ASTNode
    index: ASTNode

    def _get_type(self) -> SimileType:
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

    def _get_type(self) -> SimileType:
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

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Return(ASTNode):
    value: ASTNode | None_

    def _get_type(self) -> SimileType:
        return self.value.get_type


@dataclass(eq=False)
class ControlFlowStmt(InheritedEqMixin, ASTNode):
    op_type: ControlFlowOperator

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Statements(ASTNode):
    items: list[ASTNode]

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Else(ASTNode):
    body: ASTNode | Statements

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class If(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Elif | Else | None_ = field(default_factory=None_)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._rewrite_generators: list[BinaryOp] | None = None

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Elif(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements
    else_body: Elif | Else | None_ = field(default_factory=None_)

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class For(ASTNode):
    iterable_names: IdentList
    iterable: ASTNode
    body: ASTNode | Statements

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_

    @property
    def bound(self) -> set[Identifier]:
        return self.iterable_names.free


@dataclass
class While(ASTNode):
    condition: ASTNode
    body: ASTNode | Statements

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class StructDef(ASTNode):
    name: Identifier
    items: list[TypedName]

    def _get_type(self) -> SimileType:
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

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class ImportAll(ASTNode):
    pass

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Import(ASTNode):
    module_file_path: str
    import_objects: IdentList | None_ | ImportAll

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


@dataclass
class Start(ASTNode):
    body: Statements | None_

    def _get_type(self) -> SimileType:
        return BaseSimileType.None_


Literal = Int | Float | String | True_ | False_ | None_
Predicate = Quantifier | BinaryOp | UnaryOp | True_ | False_
Primary = StructAccess | Call | Image | Literal | Enumeration | Quantifier | Identifier
Expr = LambdaDef | Quantifier | Predicate | BinaryOp | UnaryOp | ListOp | Primary | Identifier
SimpleStmt = Expr | Assignment | ControlFlowStmt | Import
CompoundStmt = If | For | StructDef | ProcedureDef
