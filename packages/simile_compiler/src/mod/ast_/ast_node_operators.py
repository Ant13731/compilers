from __future__ import annotations
from enum import Enum, auto
from typing import TypeGuard, Literal


class BinaryOperator(Enum):
    """All binary operators in Simile (except for relation type operators)."""

    # Bools
    IMPLIES = auto()
    REV_IMPLIES = auto()
    EQUIVALENT = auto()
    NOT_EQUIVALENT = auto()
    # Numbers
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    EXPONENT = auto()
    # Num-to-bool operators
    LESS_THAN = auto()
    LESS_THAN_OR_EQUAL = auto()
    GREATER_THAN = auto()
    GREATER_THAN_OR_EQUAL = auto()
    # Equality
    EQUAL = auto()
    NOT_EQUAL = auto()
    IS = auto()
    IS_NOT = auto()
    # Set operators
    IN = auto()
    NOT_IN = auto()
    UNION = auto()
    INTERSECTION = auto()
    DIFFERENCE = auto()
    # Set-to-bool operators
    SUBSET = auto()
    SUBSET_EQ = auto()
    SUPERSET = auto()
    SUPERSET_EQ = auto()
    NOT_SUBSET = auto()
    NOT_SUBSET_EQ = auto()
    NOT_SUPERSET = auto()
    NOT_SUPERSET_EQ = auto()
    # Relation operators
    MAPLET = auto()
    RELATION_OVERRIDING = auto()
    COMPOSITION = auto()
    CARTESIAN_PRODUCT = auto()
    UPTO = auto()
    # Relation/Set operations
    DOMAIN_SUBTRACTION = auto()
    DOMAIN_RESTRICTION = auto()
    RANGE_SUBTRACTION = auto()
    RANGE_RESTRICTION = auto()


class RelationOperator(Enum):
    """All relation type binary operators in Simile."""

    RELATION = auto()
    TOTAL_RELATION = auto()
    SURJECTIVE_RELATION = auto()
    TOTAL_SURJECTIVE_RELATION = auto()
    PARTIAL_FUNCTION = auto()
    TOTAL_FUNCTION = auto()
    PARTIAL_INJECTION = auto()
    TOTAL_INJECTION = auto()
    PARTIAL_SURJECTION = auto()
    TOTAL_SURJECTION = auto()
    BIJECTION = auto()


class UnaryOperator(Enum):
    """All unary operators in Simile."""

    NOT = auto()
    NEGATIVE = auto()
    POWERSET = auto()
    NONEMPTY_POWERSET = auto()
    INVERSE = auto()


class ListOperator(Enum):
    """And/Or operators"""

    AND = auto()
    OR = auto()


class QuantifierOperator(Enum):
    FORALL = auto()
    EXISTS = auto()

    UNION_ALL = auto()
    INTERSECTION_ALL = auto()

    SUM = auto()
    PRODUCT = auto()

    SEQUENCE = auto()
    SET = auto()
    RELATION = auto()
    BAG = auto()

    def is_bool_quantifier(self) -> bool:
        """Check if the quantifier operator is a boolean quantifier (FORALL or EXISTS)."""
        return self in {QuantifierOperator.FORALL, QuantifierOperator.EXISTS}

    def is_collection_operator(self) -> bool:
        """Check if the quantifier operator is a collection operator (SEQUENCE, SET, RELATION, BAG)."""
        return self in {
            QuantifierOperator.SEQUENCE,
            QuantifierOperator.SET,
            QuantifierOperator.RELATION,
            QuantifierOperator.BAG,
        }

    def is_numerical_quantifier(self) -> bool:
        """Check if the quantifier operator is a numerical quantifier (SUM or PRODUCT)."""
        return self in {QuantifierOperator.SUM, QuantifierOperator.PRODUCT}

    def is_general_collection_operator(self) -> bool:
        """Check if the quantifier operator is a general collection operator (UNION_ALL or INTERSECTION_ALL)."""
        return self in {
            QuantifierOperator.UNION_ALL,
            QuantifierOperator.INTERSECTION_ALL,
        }

    @classmethod
    def from_collection_operator(cls, other: CollectionOperator) -> QuantifierOperator | None:
        match other:
            case CollectionOperator.SEQUENCE:
                return QuantifierOperator.SEQUENCE
            case CollectionOperator.SET:
                return QuantifierOperator.SET
            case CollectionOperator.RELATION:
                return QuantifierOperator.RELATION
            case CollectionOperator.BAG:
                return QuantifierOperator.BAG
            case _:
                return None


class ControlFlowOperator(Enum):
    """Control flow operators in Simile."""

    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()


class CollectionOperator(Enum):
    """Collection operators in Simile."""

    SEQUENCE = auto()
    SET = auto()
    RELATION = auto()
    BAG = auto()


Operators = BinaryOperator | RelationOperator | UnaryOperator | ListOperator | QuantifierOperator | ControlFlowOperator
"""Type alias for all operator enums in Simile."""
