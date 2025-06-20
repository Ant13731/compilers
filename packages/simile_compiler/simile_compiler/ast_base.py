from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from typing import TypeVar, Callable, Any
from enum import Enum, auto

T = TypeVar("T")
V = TypeVar("V")


def dataclass_traverse(
    traversal_target: Any,
    visit: Callable[[Any], T],
    visit_leaves: bool = False,
) -> list[T]:
    accumulator: list[T] = []
    _dataclass_traverse_helper(traversal_target, visit, visit_leaves, accumulator)
    return accumulator


def _dataclass_traverse_helper(
    traversal_target: Any,
    visit: Callable[[Any], T],
    visit_leaves: bool,
    accumulator: list[T],
) -> None:
    assert is_dataclass(traversal_target), "Traversal techniques only work with the dataclass `fields` function"
    accumulator.append(visit(traversal_target))
    for f in fields(traversal_target):
        field_value = getattr(traversal_target, f.name)
        if isinstance(field_value, list):
            for item in field_value:
                if is_dataclass(item):
                    _dataclass_traverse_helper(item, visit, visit_leaves, accumulator)
        elif is_dataclass(field_value):
            _dataclass_traverse_helper(field_value, visit, visit_leaves, accumulator)
        elif visit_leaves:
            accumulator.append(visit(field_value))


def find_and_replace(
    traversal_target: Any,
    rewrite_func: Callable[[Any], Any | None],
) -> Any:
    assert is_dataclass(traversal_target), "Traversal techniques only work with the dataclass `fields` function"
    # Bottom up traversal
    for f in fields(traversal_target):
        field_value = getattr(traversal_target, f.name)
        if isinstance(field_value, list):
            new_list = []
            for item in field_value:
                if is_dataclass(item):
                    new_item = find_and_replace(item, rewrite_func)
                    if new_item is not None:
                        new_list.append(new_item)
                else:
                    new_list.append(item)
            setattr(traversal_target, f.name, new_list)
        elif is_dataclass(field_value):
            new_value = find_and_replace(field_value, rewrite_func)
            setattr(traversal_target, f.name, new_value)

    replacement_target = rewrite_func(traversal_target)
    if replacement_target is not None:
        return replacement_target
    else:
        return traversal_target


# @dataclass(kw_only=True)
# class NewBoundVariableMixin:
#     binds: set[Identifier] = field(default_factory=set)
#     with_free: set[Identifier] = field(default_factory=set)


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    def well_formed(self) -> bool:
        return True

    @property
    def bound(self) -> set[Identifier]:
        """Returns the set of bound variables in the AST node."""
        return set()

    @property
    def free(self) -> set[Identifier]:
        """Returns the set of free variables in the AST node."""
        return set()

    def contains(self, node: type[ASTNode]) -> bool:
        """Check if the AST node contains a specific type of node."""
        return any(dataclass_traverse(self, lambda n: isinstance(n, node)))

        # if isinstance(self, node):
        #     return True
        # for f in fields(self):
        #     field_value = getattr(self, f.name)
        #     if isinstance(field_value, list):
        #         for item in field_value:
        #             if isinstance(item, ASTNode) and item.contains(node):
        #                 return True
        #     elif isinstance(field_value, ASTNode) and field_value.contains(node):
        #         return True
        # return False

    def find_all_instances(self, type_: type[T]) -> list[T]:
        """Returns a flattened list of all instances of a specific type in the AST.

        Most useful for finding identifiers nested within expressions.
        """
        return list(filter(None, dataclass_traverse(self, lambda n: n if isinstance(n, type_) else None)))
        # if isinstance(self, type_):
        #     return [self]
        # ret = []
        # for f in fields(self):
        #     field_value = getattr(self, f.name)
        #     if isinstance(field_value, list):
        #         for item in field_value:
        #             if isinstance(item, ASTNode):
        #                 ret += item.find_all_instances(type_)
        #     elif isinstance(field_value, ASTNode):
        #         ret += field_value.find_all_instances(type_)
        # return ret

    def pretty_print(
        self,
        ignore_fields: list[str] | None = None,
        indent=2,
    ) -> str:
        if ignore_fields is not None and self.__class__.__name__ in ignore_fields:
            indent_ = ""
            indent -= 2
            ret = ""
        else:
            # if isinstance(self, PrimitiveLiteral | Identifier):  # type: ignore # noqa
            #     if isinstance(self, None_):  # type: ignore # noqa
            #         ret = f"{self.__class__.__name__}\n"
            #         return ret
            #     ret = f"{self.__class__.__name__}: {self.value}\n"
            #     return ret

            ret = f"{self.__class__.__name__}:\n"
            indent_ = indent * " "

        for f in fields(self):
            field_value = getattr(self, f.name)

            if isinstance(field_value, ASTNode):
                if len(fields(self)) == 1:
                    ret += f"{indent_}{field_value.pretty_print(ignore_fields, indent+2)}"
                else:
                    ret += f"{indent_}{f.name}={field_value.pretty_print(ignore_fields,indent + 2)}"
                continue

            if isinstance(field_value, list):
                ret += f"{indent_}[\n"
                for item in field_value:
                    if isinstance(item, ASTNode):
                        ret += f"{indent_}{item.pretty_print(ignore_fields,indent + 2)}"
                    else:
                        ret += f"{indent_}    {item}\n"
                ret += f"{indent_}]\n"
                continue

            ret += f"{indent_}{f.name}**: {field_value}\n"
        return ret


@dataclass
class Identifier(ASTNode):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identifier):
            return False
        return self.name == other.name

    @property
    def free(self) -> set[Identifier]:
        return {self}


# Properties to determine node operations


class BinaryOpType(Enum):
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


class RelationTypes(Enum):
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


class UnaryOpType(Enum):
    NOT = auto()
    NEGATIVE = auto()
    POWERSET = auto()
    NONEMPTY_POWERSET = auto()
    INVERSE = auto()


class ListBoolType(Enum):
    AND = auto()
    OR = auto()


class BoolQuantifierType(Enum):
    FORALL = auto()
    EXISTS = auto()


class QuantifierType(Enum):
    UNION_ALL = auto()
    INTERSECTION_ALL = auto()


class ControlFlowType(Enum):
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()


class CollectionType(Enum):
    SEQUENCE = auto()
    SET = auto()
    RELATION = auto()
    BAG = auto()


# TODO:
# 1. Finish grammar and parser
# 2. Implement a TRS, based on TRAAT textbook (pg 80)
#
#
# 3. Compare language:
# - want better ease-of-use than python or haskell (maybe rust?)
# - want better performance than hand-optimized C, rust
# - see if our compiler does better than AI generated code
# - Should implement examples in python and haskell, see if they are good enough for testing, maybe find some benchmarks
