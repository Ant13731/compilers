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
from src.mod.ast_.ast_node_operators import (
    BinaryOperator,
    RelationOperator,
    ListOperator,
    UnaryOperator,
    BoolQuantifierOperator,
    QuantifierOperator,
    ControlFlowOperator,
    CollectionOperator,
)


def flatten_and_join(obj_lst: list[Any], type_: ListOperator) -> ListOp:
    """Flatten a list of objects and join them with the given ListOp."""
    flattened_objs = []
    for obj in obj_lst:
        if isinstance(obj, ListOp) and obj.op_type == type_:
            flattened_objs += obj.items
        else:
            flattened_objs.append(obj)
    return ListOp(items=flattened_objs, op_type=type_)


Implies = BinaryOp.construct_with_op(BinaryOperator.IMPLIES)
RevImplies = BinaryOp.construct_with_op(BinaryOperator.REV_IMPLIES)
Equivalent = BinaryOp.construct_with_op(BinaryOperator.EQUIVALENT)
NotEquivalent = BinaryOp.construct_with_op(BinaryOperator.NOT_EQUIVALENT)
#
Add = BinaryOp.construct_with_op(BinaryOperator.ADD)
Subtract = BinaryOp.construct_with_op(BinaryOperator.SUBTRACT)
Multiply = BinaryOp.construct_with_op(BinaryOperator.MULTIPLY)
Divide = BinaryOp.construct_with_op(BinaryOperator.DIVIDE)
Modulo = BinaryOp.construct_with_op(BinaryOperator.MODULO)
Exponent = BinaryOp.construct_with_op(BinaryOperator.EXPONENT)
#
LessThan = BinaryOp.construct_with_op(BinaryOperator.LESS_THAN)
LessThanOrEqual = BinaryOp.construct_with_op(BinaryOperator.LESS_THAN_OR_EQUAL)
GreaterThan = BinaryOp.construct_with_op(BinaryOperator.GREATER_THAN)
GreaterThanOrEqual = BinaryOp.construct_with_op(BinaryOperator.GREATER_THAN_OR_EQUAL)
#
Equal = BinaryOp.construct_with_op(BinaryOperator.EQUAL)
NotEqual = BinaryOp.construct_with_op(BinaryOperator.NOT_EQUAL)
Is = BinaryOp.construct_with_op(BinaryOperator.IS)
IsNot = BinaryOp.construct_with_op(BinaryOperator.IS_NOT)
#
In = BinaryOp.construct_with_op(BinaryOperator.IN)
NotIn = BinaryOp.construct_with_op(BinaryOperator.NOT_IN)
Union = BinaryOp.construct_with_op(BinaryOperator.UNION)
Intersection = BinaryOp.construct_with_op(BinaryOperator.INTERSECTION)
Difference = BinaryOp.construct_with_op(BinaryOperator.DIFFERENCE)
#
Subset = BinaryOp.construct_with_op(BinaryOperator.SUBSET)
SubsetEq = BinaryOp.construct_with_op(BinaryOperator.SUBSET_EQ)
SuperSet = BinaryOp.construct_with_op(BinaryOperator.SUPERSET)
SuperSetEq = BinaryOp.construct_with_op(BinaryOperator.SUPERSET_EQ)
NotSubset = BinaryOp.construct_with_op(BinaryOperator.NOT_SUBSET)
NotSubsetEq = BinaryOp.construct_with_op(BinaryOperator.NOT_SUBSET_EQ)
NotSuperSet = BinaryOp.construct_with_op(BinaryOperator.NOT_SUPERSET)
NotSuperSetEq = BinaryOp.construct_with_op(BinaryOperator.NOT_SUPERSET_EQ)
#
Maplet = BinaryOp.construct_with_op(BinaryOperator.MAPLET)
RelationOverriding = BinaryOp.construct_with_op(BinaryOperator.RELATION_OVERRIDING)
Composition = BinaryOp.construct_with_op(BinaryOperator.COMPOSITION)
CartesianProduct = BinaryOp.construct_with_op(BinaryOperator.CARTESIAN_PRODUCT)
UpTo = BinaryOp.construct_with_op(BinaryOperator.UPTO)
#
DomainSubtraction = BinaryOp.construct_with_op(BinaryOperator.DOMAIN_SUBTRACTION)
DomainRestriction = BinaryOp.construct_with_op(BinaryOperator.DOMAIN_RESTRICTION)
RangeSubtraction = BinaryOp.construct_with_op(BinaryOperator.RANGE_SUBTRACTION)
RangeRestriction = BinaryOp.construct_with_op(BinaryOperator.RANGE_RESTRICTION)
#
#
Relation = RelationOp.construct_with_op(RelationOperator.RELATION)
TotalRelation = RelationOp.construct_with_op(RelationOperator.TOTAL_RELATION)
SurjectiveRelation = RelationOp.construct_with_op(RelationOperator.SURJECTIVE_RELATION)
TotalSurjectiveRelation = RelationOp.construct_with_op(RelationOperator.TOTAL_SURJECTIVE_RELATION)
PartialFunction = RelationOp.construct_with_op(RelationOperator.PARTIAL_FUNCTION)
TotalFunction = RelationOp.construct_with_op(RelationOperator.TOTAL_FUNCTION)
PartialInjection = RelationOp.construct_with_op(RelationOperator.PARTIAL_INJECTION)
TotalInjection = RelationOp.construct_with_op(RelationOperator.TOTAL_INJECTION)
PartialSurjection = RelationOp.construct_with_op(RelationOperator.PARTIAL_SURJECTION)
TotalSurjection = RelationOp.construct_with_op(RelationOperator.TOTAL_SURJECTION)
Bijection = RelationOp.construct_with_op(RelationOperator.BIJECTION)
#
#
Not = UnaryOp.construct_with_op(UnaryOperator.NOT)
Negative = UnaryOp.construct_with_op(UnaryOperator.NEGATIVE)
Powerset = UnaryOp.construct_with_op(UnaryOperator.POWERSET)
NonemptyPowerset = UnaryOp.construct_with_op(UnaryOperator.NONEMPTY_POWERSET)
Inverse = UnaryOp.construct_with_op(UnaryOperator.INVERSE)
#
#
And = ListOp.construct_with_op(ListOperator.AND)
Or = ListOp.construct_with_op(ListOperator.OR)
#
#
Forall = BoolQuantifier.construct_with_op(BoolQuantifierOperator.FORALL)
Exists = BoolQuantifier.construct_with_op(BoolQuantifierOperator.EXISTS)
#
#
UnionAll = Quantifier.construct_with_op(QuantifierOperator.UNION_ALL)
IntersectionAll = Quantifier.construct_with_op(QuantifierOperator.INTERSECTION_ALL)
#
#
Break = lambda: ControlFlowStmt(ControlFlowOperator.BREAK)
Continue = lambda: ControlFlowStmt(ControlFlowOperator.CONTINUE)
Pass = lambda: ControlFlowStmt(ControlFlowOperator.PASS)
#
#
SequenceEnumeration = Enumeration.construct_with_op(CollectionOperator.SEQUENCE)
SetEnumeration = Enumeration.construct_with_op(CollectionOperator.SET)
RelationEnumeration = Enumeration.construct_with_op(CollectionOperator.RELATION)
BagEnumeration = Enumeration.construct_with_op(CollectionOperator.BAG)
#
#
SequenceComprehension = Comprehension.construct_with_op(CollectionOperator.SEQUENCE)
SetComprehension = Comprehension.construct_with_op(CollectionOperator.SET)
RelationComprehension = Comprehension.construct_with_op(CollectionOperator.RELATION)
BagComprehension = Comprehension.construct_with_op(CollectionOperator.BAG)
