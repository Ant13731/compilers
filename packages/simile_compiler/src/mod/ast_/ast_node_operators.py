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

    def is_total(self) -> bool:
        """Check if the relation operator is total on the domain."""
        return self in {
            RelationOperator.TOTAL_RELATION,
            RelationOperator.TOTAL_SURJECTIVE_RELATION,
            RelationOperator.TOTAL_FUNCTION,
            RelationOperator.TOTAL_INJECTION,
            RelationOperator.TOTAL_SURJECTION,
            RelationOperator.BIJECTION,
        }

    def is_surjective(self) -> bool:
        """Check if the relation operator is total on the range (onto)."""
        return self in {
            RelationOperator.SURJECTIVE_RELATION,
            RelationOperator.TOTAL_SURJECTIVE_RELATION,
            RelationOperator.PARTIAL_SURJECTION,
            RelationOperator.TOTAL_SURJECTION,
            RelationOperator.BIJECTION,
        }

    def is_injective(self) -> bool:
        """Check if the relation operator is injective (one-to-one)."""
        return self in {
            RelationOperator.PARTIAL_INJECTION,
            RelationOperator.TOTAL_INJECTION,
            RelationOperator.BIJECTION,
        }

    def is_function(self) -> bool:
        """Check if the relation operator is a function (partial or total)."""
        return self in {
            RelationOperator.PARTIAL_FUNCTION,
            RelationOperator.TOTAL_FUNCTION,
            RelationOperator.PARTIAL_INJECTION,
            RelationOperator.TOTAL_INJECTION,
            RelationOperator.PARTIAL_SURJECTION,
            RelationOperator.TOTAL_SURJECTION,
            RelationOperator.BIJECTION,
        }

    def remove_total(self) -> RelationOperator:
        match self:
            case RelationOperator.TOTAL_RELATION:
                return RelationOperator.RELATION
            case RelationOperator.TOTAL_SURJECTIVE_RELATION:
                return RelationOperator.SURJECTIVE_RELATION
            case RelationOperator.TOTAL_FUNCTION:
                return RelationOperator.PARTIAL_FUNCTION
            case RelationOperator.TOTAL_INJECTION:
                return RelationOperator.PARTIAL_INJECTION
            case RelationOperator.TOTAL_SURJECTION:
                return RelationOperator.PARTIAL_SURJECTION
            case RelationOperator.BIJECTION:
                # No longer surjective since previous one-to-one correspondence loses the elements making the function onto
                # Still injective since still one-to-one
                return RelationOperator.PARTIAL_INJECTION
            case _:
                return self

    def remove_surjective(self) -> RelationOperator:
        match self:
            case RelationOperator.SURJECTIVE_RELATION:
                return RelationOperator.RELATION
            case RelationOperator.TOTAL_SURJECTIVE_RELATION:
                return RelationOperator.TOTAL_RELATION
            case RelationOperator.PARTIAL_SURJECTION:
                return RelationOperator.PARTIAL_FUNCTION
            case RelationOperator.TOTAL_SURJECTION:
                return RelationOperator.TOTAL_FUNCTION
            case RelationOperator.BIJECTION:
                # No longer surjective since previous one-to-one correspondence loses the elements making the function onto
                return RelationOperator.PARTIAL_INJECTION
            case _:
                return self

    def remove_injective(self) -> RelationOperator:
        match self:
            case RelationOperator.PARTIAL_INJECTION:
                return RelationOperator.PARTIAL_FUNCTION
            case RelationOperator.TOTAL_INJECTION:
                return RelationOperator.TOTAL_FUNCTION
            case RelationOperator.BIJECTION:
                return RelationOperator.TOTAL_SURJECTION
            case _:
                return self

    def remove_function(self) -> RelationOperator:
        """Remove the function property from the relation operator."""
        match self:
            case RelationOperator.PARTIAL_FUNCTION | RelationOperator.PARTIAL_INJECTION:
                return RelationOperator.RELATION
            case RelationOperator.TOTAL_FUNCTION | RelationOperator.TOTAL_INJECTION:
                return RelationOperator.TOTAL_RELATION
            case RelationOperator.PARTIAL_SURJECTION:
                return RelationOperator.SURJECTIVE_RELATION
            case RelationOperator.TOTAL_SURJECTION | RelationOperator.BIJECTION:
                return RelationOperator.TOTAL_SURJECTIVE_RELATION
            case _:
                return self

    def make_function(self) -> RelationOperator:
        match self:
            case RelationOperator.RELATION:
                return RelationOperator.PARTIAL_FUNCTION
            case RelationOperator.TOTAL_RELATION:
                return RelationOperator.TOTAL_FUNCTION
            case RelationOperator.SURJECTIVE_RELATION:
                return RelationOperator.PARTIAL_SURJECTION
            case RelationOperator.TOTAL_SURJECTIVE_RELATION:
                return RelationOperator.TOTAL_SURJECTION
            case _:
                return self

    def make_total(self) -> RelationOperator:
        """Make the relation operator total on the domain."""
        match self:
            case RelationOperator.RELATION:
                return RelationOperator.TOTAL_RELATION
            case RelationOperator.SURJECTIVE_RELATION:
                return RelationOperator.TOTAL_SURJECTIVE_RELATION
            case RelationOperator.PARTIAL_FUNCTION:
                return RelationOperator.TOTAL_FUNCTION
            case RelationOperator.PARTIAL_INJECTION:
                return RelationOperator.TOTAL_INJECTION
            case RelationOperator.PARTIAL_SURJECTION:
                return RelationOperator.TOTAL_SURJECTION
            case _:
                return self

    def make_surjective(self) -> RelationOperator:
        """Make the relation operator surjective (onto)."""
        match self:
            case RelationOperator.RELATION:
                return RelationOperator.SURJECTIVE_RELATION
            case RelationOperator.TOTAL_RELATION:
                return RelationOperator.TOTAL_SURJECTIVE_RELATION
            case RelationOperator.PARTIAL_FUNCTION:
                return RelationOperator.PARTIAL_SURJECTION
            case RelationOperator.TOTAL_FUNCTION:
                return RelationOperator.TOTAL_SURJECTION
            case RelationOperator.TOTAL_INJECTION:
                return RelationOperator.BIJECTION
            case _:
                return self

    def make_injective(self) -> RelationOperator:
        """Make the relation operator injective (one-to-one). Input must be a function"""
        if not self.is_function():
            raise ValueError(f"Cannot make {self.name} injective, it is not a function (call make_function first).")

        match self:
            case RelationOperator.PARTIAL_FUNCTION:
                return RelationOperator.PARTIAL_INJECTION
            case RelationOperator.TOTAL_FUNCTION:
                return RelationOperator.TOTAL_INJECTION
            # missing info for partial bijection?
            case RelationOperator.TOTAL_SURJECTION:
                return RelationOperator.BIJECTION
            case _:
                return self

    def inverse(self) -> RelationOperator:
        # Criteria:
        # If anything is total, the reverse will be surjective
        # Functions are demoted to relations
        # If anything is surjective, the reverse will be total
        # If anything is injective, the reverse will be injective

        match self:
            case RelationOperator.RELATION:
                return RelationOperator.RELATION
            case RelationOperator.TOTAL_RELATION:
                return RelationOperator.SURJECTIVE_RELATION
            case RelationOperator.SURJECTIVE_RELATION:
                return RelationOperator.TOTAL_RELATION
            case RelationOperator.TOTAL_SURJECTIVE_RELATION:
                return RelationOperator.TOTAL_SURJECTIVE_RELATION

            case RelationOperator.PARTIAL_FUNCTION:
                return RelationOperator.RELATION
            case RelationOperator.TOTAL_FUNCTION:
                return RelationOperator.SURJECTIVE_RELATION
            case RelationOperator.PARTIAL_INJECTION:
                return RelationOperator.PARTIAL_INJECTION
            case RelationOperator.TOTAL_INJECTION:
                # FIXME: isnt this a partial bijection? it would be surjective but not total
                return RelationOperator.PARTIAL_INJECTION

            case RelationOperator.PARTIAL_SURJECTION:
                return RelationOperator.TOTAL_RELATION
            case RelationOperator.TOTAL_SURJECTION:
                return RelationOperator.TOTAL_SURJECTIVE_RELATION
            case RelationOperator.BIJECTION:
                return RelationOperator.BIJECTION

    def get_resulting_operator(self, other: RelationOperator, combining_operator: BinaryOperator) -> RelationOperator:
        match combining_operator:
            case BinaryOperator.RELATION_OVERRIDING:

                ret = RelationOperator.RELATION
                if self.is_function() and other.is_function():
                    ret = ret.make_function()
                if other.is_total():
                    ret = ret.make_total()
                if other.is_surjective():
                    ret = ret.make_surjective()
                return ret

            case BinaryOperator.COMPOSITION:
                if self == RelationOperator.BIJECTION and other == RelationOperator.BIJECTION:
                    return RelationOperator.BIJECTION

                ret = RelationOperator.RELATION
                if self.is_function() and other.is_function():
                    ret = ret.make_function()
                if self.is_total():
                    ret = ret.make_total()
                if self.is_injective() and other.is_injective():
                    ret = ret.make_injective()
                return ret

            case _:
                raise ValueError(f"Cannot combine relation operators with {combining_operator.name} operator (types to be combined were: {self.name}, {other.name})")

    def get_resulting_operator_set_or_unary(self, combining_operator: BinaryOperator | UnaryOperator) -> RelationOperator:
        match combining_operator:
            case BinaryOperator.DOMAIN_RESTRICTION | BinaryOperator.DOMAIN_SUBTRACTION:
                return self.remove_total()
            case BinaryOperator.RANGE_RESTRICTION | BinaryOperator.RANGE_SUBTRACTION:
                return self.remove_surjective()
            case UnaryOperator.INVERSE:
                return self.inverse()
            case _:
                return self


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
