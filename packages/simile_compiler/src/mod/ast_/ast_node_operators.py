from __future__ import annotations
from enum import Enum, auto
from typing import TypeGuard, Literal


class BinaryOperator(Enum):
    """All binary operators in Simile (except for relation type operators)."""

    # Bools
    IMPLIES = auto()
    EQUIVALENT = auto()
    NOT_EQUIVALENT = auto()
    # Numbers
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    INT_DIVIDE = auto()
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
    CONCAT = auto()

    def pretty_print(self) -> str:
        pretty_print_lookup = {
            BinaryOperator.IMPLIES: "⇒",
            BinaryOperator.EQUIVALENT: "≡",
            BinaryOperator.NOT_EQUIVALENT: "≢",
            #: ,
            BinaryOperator.ADD: "+",
            BinaryOperator.SUBTRACT: "-",
            BinaryOperator.MULTIPLY: "*",
            BinaryOperator.DIVIDE: "/",
            BinaryOperator.INT_DIVIDE: "div",
            BinaryOperator.MODULO: "mod",
            BinaryOperator.EXPONENT: "^",
            #: ,
            BinaryOperator.LESS_THAN: "<",
            BinaryOperator.LESS_THAN_OR_EQUAL: "≤",
            BinaryOperator.GREATER_THAN: ">",
            BinaryOperator.GREATER_THAN_OR_EQUAL: "≥",
            #: ,
            BinaryOperator.EQUAL: "=",
            BinaryOperator.NOT_EQUAL: "≠",
            BinaryOperator.IS: "is",
            BinaryOperator.IS_NOT: "is not",
            #: ,
            BinaryOperator.IN: "∈",
            BinaryOperator.NOT_IN: "∉",
            BinaryOperator.UNION: "∪",
            BinaryOperator.INTERSECTION: "∩",
            BinaryOperator.DIFFERENCE: "∖",
            #: ,
            BinaryOperator.SUBSET: "⊂",
            BinaryOperator.SUBSET_EQ: "⊆",
            BinaryOperator.SUPERSET: "⊃",
            BinaryOperator.SUPERSET_EQ: "⊇",
            BinaryOperator.NOT_SUBSET: "⊄",
            BinaryOperator.NOT_SUBSET_EQ: "⊈",
            BinaryOperator.NOT_SUPERSET: "⊅",
            BinaryOperator.NOT_SUPERSET_EQ: "⊉",
            #: ,
            BinaryOperator.MAPLET: "↦",
            BinaryOperator.RELATION_OVERRIDING: "⊕",
            BinaryOperator.COMPOSITION: "∘",
            BinaryOperator.CARTESIAN_PRODUCT: "×",
            BinaryOperator.UPTO: "..",
            #: ,
            BinaryOperator.DOMAIN_SUBTRACTION: "◁",
            BinaryOperator.DOMAIN_RESTRICTION: "⩤",
            BinaryOperator.RANGE_SUBTRACTION: "▷",
            BinaryOperator.RANGE_RESTRICTION: "⩥",
            #: ,
            BinaryOperator.CONCAT: "⧺",
        }
        return pretty_print_lookup.get(self, self.name)


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

    # See RelationSubTypeMask
    # def is_total(self) -> bool:
    #     """Check if the relation operator is total on the domain."""
    #     return self in {
    #         RelationOperator.TOTAL_RELATION,
    #         RelationOperator.TOTAL_SURJECTIVE_RELATION,
    #         RelationOperator.TOTAL_FUNCTION,
    #         RelationOperator.TOTAL_INJECTION,
    #         RelationOperator.TOTAL_SURJECTION,
    #         RelationOperator.BIJECTION,
    #     }

    # def is_surjective(self) -> bool:
    #     """Check if the relation operator is total on the range (onto)."""
    #     return self in {
    #         RelationOperator.SURJECTIVE_RELATION,
    #         RelationOperator.TOTAL_SURJECTIVE_RELATION,
    #         RelationOperator.PARTIAL_SURJECTION,
    #         RelationOperator.TOTAL_SURJECTION,
    #         RelationOperator.BIJECTION,
    #     }

    # def is_injective(self) -> bool:
    #     """Check if the relation operator is injective (one-to-one)."""
    #     return self in {
    #         RelationOperator.PARTIAL_INJECTION,
    #         RelationOperator.TOTAL_INJECTION,
    #         RelationOperator.BIJECTION,
    #     }

    # def is_function(self) -> bool:
    #     """Check if the relation operator is a function (partial or total)."""
    #     return self in {
    #         RelationOperator.PARTIAL_FUNCTION,
    #         RelationOperator.TOTAL_FUNCTION,
    #         RelationOperator.PARTIAL_INJECTION,
    #         RelationOperator.TOTAL_INJECTION,
    #         RelationOperator.PARTIAL_SURJECTION,
    #         RelationOperator.TOTAL_SURJECTION,
    #         RelationOperator.BIJECTION,
    #     }

    # def remove_total(self) -> RelationOperator:
    #     match self:
    #         case RelationOperator.TOTAL_RELATION:
    #             return RelationOperator.RELATION
    #         case RelationOperator.TOTAL_SURJECTIVE_RELATION:
    #             return RelationOperator.SURJECTIVE_RELATION
    #         case RelationOperator.TOTAL_FUNCTION:
    #             return RelationOperator.PARTIAL_FUNCTION
    #         case RelationOperator.TOTAL_INJECTION:
    #             return RelationOperator.PARTIAL_INJECTION
    #         case RelationOperator.TOTAL_SURJECTION:
    #             return RelationOperator.PARTIAL_SURJECTION
    #         case RelationOperator.BIJECTION:
    #             # No longer surjective since previous one-to-one correspondence loses the elements making the function onto
    #             # Still injective since still one-to-one
    #             return RelationOperator.PARTIAL_INJECTION
    #         case _:
    #             return self

    # def remove_surjective(self) -> RelationOperator:
    #     match self:
    #         case RelationOperator.SURJECTIVE_RELATION:
    #             return RelationOperator.RELATION
    #         case RelationOperator.TOTAL_SURJECTIVE_RELATION:
    #             return RelationOperator.TOTAL_RELATION
    #         case RelationOperator.PARTIAL_SURJECTION:
    #             return RelationOperator.PARTIAL_FUNCTION
    #         case RelationOperator.TOTAL_SURJECTION:
    #             return RelationOperator.TOTAL_FUNCTION
    #         case RelationOperator.BIJECTION:
    #             # No longer surjective since previous one-to-one correspondence loses the elements making the function onto
    #             return RelationOperator.PARTIAL_INJECTION
    #         case _:
    #             return self

    # def remove_injective(self) -> RelationOperator:
    #     match self:
    #         case RelationOperator.PARTIAL_INJECTION:
    #             return RelationOperator.PARTIAL_FUNCTION
    #         case RelationOperator.TOTAL_INJECTION:
    #             return RelationOperator.TOTAL_FUNCTION
    #         case RelationOperator.BIJECTION:
    #             return RelationOperator.TOTAL_SURJECTION
    #         case _:
    #             return self

    # def remove_function(self) -> RelationOperator:
    #     """Remove the function property from the relation operator."""
    #     match self:
    #         case RelationOperator.PARTIAL_FUNCTION | RelationOperator.PARTIAL_INJECTION:
    #             return RelationOperator.RELATION
    #         case RelationOperator.TOTAL_FUNCTION | RelationOperator.TOTAL_INJECTION:
    #             return RelationOperator.TOTAL_RELATION
    #         case RelationOperator.PARTIAL_SURJECTION:
    #             return RelationOperator.SURJECTIVE_RELATION
    #         case RelationOperator.TOTAL_SURJECTION | RelationOperator.BIJECTION:
    #             return RelationOperator.TOTAL_SURJECTIVE_RELATION
    #         case _:
    #             return self

    # def make_function(self) -> RelationOperator:
    #     match self:
    #         case RelationOperator.RELATION:
    #             return RelationOperator.PARTIAL_FUNCTION
    #         case RelationOperator.TOTAL_RELATION:
    #             return RelationOperator.TOTAL_FUNCTION
    #         case RelationOperator.SURJECTIVE_RELATION:
    #             return RelationOperator.PARTIAL_SURJECTION
    #         case RelationOperator.TOTAL_SURJECTIVE_RELATION:
    #             return RelationOperator.TOTAL_SURJECTION
    #         case _:
    #             return self

    # def make_total(self) -> RelationOperator:
    #     """Make the relation operator total on the domain."""
    #     match self:
    #         case RelationOperator.RELATION:
    #             return RelationOperator.TOTAL_RELATION
    #         case RelationOperator.SURJECTIVE_RELATION:
    #             return RelationOperator.TOTAL_SURJECTIVE_RELATION
    #         case RelationOperator.PARTIAL_FUNCTION:
    #             return RelationOperator.TOTAL_FUNCTION
    #         case RelationOperator.PARTIAL_INJECTION:
    #             return RelationOperator.TOTAL_INJECTION
    #         case RelationOperator.PARTIAL_SURJECTION:
    #             return RelationOperator.TOTAL_SURJECTION
    #         case _:
    #             return self

    # def make_surjective(self) -> RelationOperator:
    #     """Make the relation operator surjective (onto)."""
    #     match self:
    #         case RelationOperator.RELATION:
    #             return RelationOperator.SURJECTIVE_RELATION
    #         case RelationOperator.TOTAL_RELATION:
    #             return RelationOperator.TOTAL_SURJECTIVE_RELATION
    #         case RelationOperator.PARTIAL_FUNCTION:
    #             return RelationOperator.PARTIAL_SURJECTION
    #         case RelationOperator.TOTAL_FUNCTION:
    #             return RelationOperator.TOTAL_SURJECTION
    #         case RelationOperator.TOTAL_INJECTION:
    #             return RelationOperator.BIJECTION
    #         case _:
    #             return self

    # def make_injective(self) -> RelationOperator:
    #     """Make the relation operator injective (one-to-one). Input must be a function"""
    #     if not self.is_function():
    #         raise ValueError(f"Cannot make {self.name} injective, it is not a function (call make_function first).")

    #     match self:
    #         case RelationOperator.PARTIAL_FUNCTION:
    #             return RelationOperator.PARTIAL_INJECTION
    #         case RelationOperator.TOTAL_FUNCTION:
    #             return RelationOperator.TOTAL_INJECTION
    #         # missing info for partial bijection?
    #         case RelationOperator.TOTAL_SURJECTION:
    #             return RelationOperator.BIJECTION
    #         case _:
    #             return self

    # def inverse(self) -> RelationOperator:
    #     # Criteria:
    #     # If anything is total, the reverse will be surjective
    #     # Functions are demoted to relations
    #     # If anything is surjective, the reverse will be total
    #     # If anything is injective, the reverse will be injective

    #     match self:
    #         case RelationOperator.RELATION:
    #             return RelationOperator.RELATION
    #         case RelationOperator.TOTAL_RELATION:
    #             return RelationOperator.SURJECTIVE_RELATION
    #         case RelationOperator.SURJECTIVE_RELATION:
    #             return RelationOperator.TOTAL_RELATION
    #         case RelationOperator.TOTAL_SURJECTIVE_RELATION:
    #             return RelationOperator.TOTAL_SURJECTIVE_RELATION

    #         case RelationOperator.PARTIAL_FUNCTION:
    #             return RelationOperator.RELATION
    #         case RelationOperator.TOTAL_FUNCTION:
    #             return RelationOperator.SURJECTIVE_RELATION
    #         case RelationOperator.PARTIAL_INJECTION:
    #             return RelationOperator.PARTIAL_INJECTION
    #         case RelationOperator.TOTAL_INJECTION:
    #             # FIXME: isnt this a partial bijection? it would be surjective but not total
    #             return RelationOperator.PARTIAL_INJECTION

    #         case RelationOperator.PARTIAL_SURJECTION:
    #             return RelationOperator.TOTAL_RELATION
    #         case RelationOperator.TOTAL_SURJECTION:
    #             return RelationOperator.TOTAL_SURJECTIVE_RELATION
    #         case RelationOperator.BIJECTION:
    #             return RelationOperator.BIJECTION

    def pretty_print(self) -> str:
        pretty_print_lookup = {
            RelationOperator.RELATION: "↔",
            RelationOperator.TOTAL_RELATION: "<<->",
            RelationOperator.SURJECTIVE_RELATION: "<->>",
            RelationOperator.TOTAL_SURJECTIVE_RELATION: "<<->>",
            RelationOperator.PARTIAL_FUNCTION: "⇸",
            RelationOperator.TOTAL_FUNCTION: "→",
            RelationOperator.PARTIAL_INJECTION: "⤔",
            RelationOperator.TOTAL_INJECTION: "↣",
            RelationOperator.PARTIAL_SURJECTION: "⤀",
            RelationOperator.TOTAL_SURJECTION: "↠",
            RelationOperator.BIJECTION: "⤖",
        }
        return pretty_print_lookup.get(self, self.name)


class UnaryOperator(Enum):
    """All unary operators in Simile."""

    NOT = auto()
    NEGATIVE = auto()
    POWERSET = auto()
    NONEMPTY_POWERSET = auto()
    INVERSE = auto()

    def pretty_print(self) -> str:
        pretty_print_lookup = {
            UnaryOperator.NOT: "¬",
            UnaryOperator.NEGATIVE: "-",
            UnaryOperator.POWERSET: "ℙ",
            UnaryOperator.NONEMPTY_POWERSET: "ℙ₁",
            UnaryOperator.INVERSE: "⁻¹",
        }
        return pretty_print_lookup.get(self, self.name)


class ListOperator(Enum):
    """And/Or operators"""

    AND = auto()
    OR = auto()

    def pretty_print(self) -> str:
        pretty_print_lookup = {
            ListOperator.AND: "∧",
            ListOperator.OR: "∨",
        }
        return pretty_print_lookup.get(self, self.name)


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

    def pretty_print(self) -> str:
        pretty_print_lookup = {
            QuantifierOperator.FORALL: "∀",
            QuantifierOperator.EXISTS: "∃",
            QuantifierOperator.UNION_ALL: "⋃",
            QuantifierOperator.INTERSECTION_ALL: "⋂",
            QuantifierOperator.SUM: "Σ",
            QuantifierOperator.PRODUCT: "Π",
            QuantifierOperator.SEQUENCE: "⟨⟩",
            QuantifierOperator.SET: "{}",
            QuantifierOperator.RELATION: "{}",
            QuantifierOperator.BAG: "⟦⟧",
        }
        return pretty_print_lookup.get(self, self.name)


class ControlFlowOperator(Enum):
    """Control flow operators in Simile."""

    BREAK = auto()
    CONTINUE = auto()
    SKIP = auto()

    def pretty_print(self) -> str:
        pretty_print_lookup = {
            ControlFlowOperator.BREAK: "break",
            ControlFlowOperator.CONTINUE: "continue",
            ControlFlowOperator.SKIP: "skip",
        }
        return pretty_print_lookup.get(self, self.name)


class CollectionOperator(Enum):
    """Collection operators in Simile."""

    SEQUENCE = auto()
    SET = auto()
    RELATION = auto()
    BAG = auto()
    TUPLE = auto()

    def pretty_print(self) -> str:
        quantifier_op_version = QuantifierOperator.from_collection_operator(self)
        return quantifier_op_version.pretty_print() if quantifier_op_version else self.name


Operators = BinaryOperator | RelationOperator | UnaryOperator | ListOperator | QuantifierOperator | ControlFlowOperator
"""Type alias for all operator enums in Simile."""
