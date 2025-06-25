from enum import Enum, auto


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


class BoolQuantifierOperator(Enum):
    """Forall/Exists operators"""

    FORALL = auto()
    EXISTS = auto()


class QuantifierOperator(Enum):
    """UnionAll/IntersectionAll operators"""

    UNION_ALL = auto()
    INTERSECTION_ALL = auto()


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


Operators = BinaryOperator | RelationOperator | UnaryOperator | ListOperator | BoolQuantifierOperator | QuantifierOperator | ControlFlowOperator
"""Type alias for all operator enums in Simile."""
