from typing import Any

from src.mod.ast_.ast_nodes import (
    BinaryOp,
    RelationOp,
    ListOp,
    UnaryOp,
    BoolQuantifier,
    Quantifier,
    ControlFlowStmt,
    Enumeration,
    Comprehension,
)
from src.mod.ast_.ast_node_types import (
    BinaryOpType,
    RelationTypes,
    ListBoolType,
    UnaryOpType,
    BoolQuantifierType,
    QuantifierType,
    ControlFlowType,
    CollectionType,
)


def flatten_and_join(obj_lst: list[Any], type_: ListBoolType) -> ListOp:  # noqa: F405
    """Flatten a list of objects and join them with the given ListOp."""
    flattened_objs = []
    for obj in obj_lst:
        if isinstance(obj, ListOp) and obj.op_type == type_:  # noqa: F405
            flattened_objs += obj.items
        else:
            flattened_objs.append(obj)
    return ListOp(items=flattened_objs, op_type=type_)  # noqa: F405


Implies = BinaryOp.construct_with_op(BinaryOpType.IMPLIES)
RevImplies = BinaryOp.construct_with_op(BinaryOpType.REV_IMPLIES)
Equivalent = BinaryOp.construct_with_op(BinaryOpType.EQUIVALENT)
NotEquivalent = BinaryOp.construct_with_op(BinaryOpType.NOT_EQUIVALENT)
#
Add = BinaryOp.construct_with_op(BinaryOpType.ADD)
Subtract = BinaryOp.construct_with_op(BinaryOpType.SUBTRACT)
Multiply = BinaryOp.construct_with_op(BinaryOpType.MULTIPLY)
Divide = BinaryOp.construct_with_op(BinaryOpType.DIVIDE)
Modulo = BinaryOp.construct_with_op(BinaryOpType.MODULO)
Exponent = BinaryOp.construct_with_op(BinaryOpType.EXPONENT)
#
LessThan = BinaryOp.construct_with_op(BinaryOpType.LESS_THAN)
LessThanOrEqual = BinaryOp.construct_with_op(BinaryOpType.LESS_THAN_OR_EQUAL)
GreaterThan = BinaryOp.construct_with_op(BinaryOpType.GREATER_THAN)
GreaterThanOrEqual = BinaryOp.construct_with_op(BinaryOpType.GREATER_THAN_OR_EQUAL)
#
Equal = BinaryOp.construct_with_op(BinaryOpType.EQUAL)
NotEqual = BinaryOp.construct_with_op(BinaryOpType.NOT_EQUAL)
Is = BinaryOp.construct_with_op(BinaryOpType.IS)
IsNot = BinaryOp.construct_with_op(BinaryOpType.IS_NOT)
#
In = BinaryOp.construct_with_op(BinaryOpType.IN)
NotIn = BinaryOp.construct_with_op(BinaryOpType.NOT_IN)
Union = BinaryOp.construct_with_op(BinaryOpType.UNION)
Intersection = BinaryOp.construct_with_op(BinaryOpType.INTERSECTION)
Difference = BinaryOp.construct_with_op(BinaryOpType.DIFFERENCE)
#
Subset = BinaryOp.construct_with_op(BinaryOpType.SUBSET)
SubsetEq = BinaryOp.construct_with_op(BinaryOpType.SUBSET_EQ)
SuperSet = BinaryOp.construct_with_op(BinaryOpType.SUPERSET)
SuperSetEq = BinaryOp.construct_with_op(BinaryOpType.SUPERSET_EQ)
NotSubset = BinaryOp.construct_with_op(BinaryOpType.NOT_SUBSET)
NotSubsetEq = BinaryOp.construct_with_op(BinaryOpType.NOT_SUBSET_EQ)
NotSuperSet = BinaryOp.construct_with_op(BinaryOpType.NOT_SUPERSET)
NotSuperSetEq = BinaryOp.construct_with_op(BinaryOpType.NOT_SUPERSET_EQ)
#
Maplet = BinaryOp.construct_with_op(BinaryOpType.MAPLET)
RelationOverriding = BinaryOp.construct_with_op(BinaryOpType.RELATION_OVERRIDING)
Composition = BinaryOp.construct_with_op(BinaryOpType.COMPOSITION)
CartesianProduct = BinaryOp.construct_with_op(BinaryOpType.CARTESIAN_PRODUCT)
UpTo = BinaryOp.construct_with_op(BinaryOpType.UPTO)
#
DomainSubtraction = BinaryOp.construct_with_op(BinaryOpType.DOMAIN_SUBTRACTION)
DomainRestriction = BinaryOp.construct_with_op(BinaryOpType.DOMAIN_RESTRICTION)
RangeSubtraction = BinaryOp.construct_with_op(BinaryOpType.RANGE_SUBTRACTION)
RangeRestriction = BinaryOp.construct_with_op(BinaryOpType.RANGE_RESTRICTION)
#
#
Relation = RelationOp.construct_with_op(RelationTypes.RELATION)
TotalRelation = RelationOp.construct_with_op(RelationTypes.TOTAL_RELATION)
SurjectiveRelation = RelationOp.construct_with_op(RelationTypes.SURJECTIVE_RELATION)
TotalSurjectiveRelation = RelationOp.construct_with_op(RelationTypes.TOTAL_SURJECTIVE_RELATION)
PartialFunction = RelationOp.construct_with_op(RelationTypes.PARTIAL_FUNCTION)
TotalFunction = RelationOp.construct_with_op(RelationTypes.TOTAL_FUNCTION)
PartialInjection = RelationOp.construct_with_op(RelationTypes.PARTIAL_INJECTION)
TotalInjection = RelationOp.construct_with_op(RelationTypes.TOTAL_INJECTION)
PartialSurjection = RelationOp.construct_with_op(RelationTypes.PARTIAL_SURJECTION)
TotalSurjection = RelationOp.construct_with_op(RelationTypes.TOTAL_SURJECTION)
Bijection = RelationOp.construct_with_op(RelationTypes.BIJECTION)
#
#
Not = UnaryOp.construct_with_op(UnaryOpType.NOT)
Negative = UnaryOp.construct_with_op(UnaryOpType.NEGATIVE)
Powerset = UnaryOp.construct_with_op(UnaryOpType.POWERSET)
NonemptyPowerset = UnaryOp.construct_with_op(UnaryOpType.NONEMPTY_POWERSET)
Inverse = UnaryOp.construct_with_op(UnaryOpType.INVERSE)
#
#
And = ListOp.construct_with_op(ListBoolType.AND)
Or = ListOp.construct_with_op(ListBoolType.OR)
#
#
Forall = BoolQuantifier.construct_with_op(BoolQuantifierType.FORALL)
Exists = BoolQuantifier.construct_with_op(BoolQuantifierType.EXISTS)
#
#
UnionAll = Quantifier.construct_with_op(QuantifierType.UNION_ALL)
IntersectionAll = Quantifier.construct_with_op(QuantifierType.INTERSECTION_ALL)
#
#
Break = lambda: ControlFlowStmt(ControlFlowType.BREAK)
Continue = lambda: ControlFlowStmt(ControlFlowType.CONTINUE)
Pass = lambda: ControlFlowStmt(ControlFlowType.PASS)
#
#
SequenceEnumeration = Enumeration.construct_with_op(CollectionType.SEQUENCE)
SetEnumeration = Enumeration.construct_with_op(CollectionType.SET)
RelationEnumeration = Enumeration.construct_with_op(CollectionType.RELATION)
BagEnumeration = Enumeration.construct_with_op(CollectionType.BAG)
#
#
SequenceComprehension = Comprehension.construct_with_op(CollectionType.SEQUENCE)
SetComprehension = Comprehension.construct_with_op(CollectionType.SET)
RelationComprehension = Comprehension.construct_with_op(CollectionType.RELATION)
BagComprehension = Comprehension.construct_with_op(CollectionType.BAG)
