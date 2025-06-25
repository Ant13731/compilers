from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Generator, Any, TypeVar

from src.mod.ast_.ast_node_operators import Operators
from src.mod.ast_.dataclass_helpers import dataclass_traverse
from src.mod.ast_.type_analysis_types import SimileType, DeferToSymbolTable

T = TypeVar("T")


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    def well_formed(self) -> bool:
        """Check if the variables in expressions are well-formed (i.e., no clashes between :attr:`bound` and :attr:`free` variables)."""
        return True

    @property
    def bound(self) -> set[Identifier]:
        """Returns the set of bound variables in the AST node."""
        return set()

    @property
    def free(self) -> set[Identifier]:
        """Returns the set of free variables in the AST node."""
        return set()

    @property
    def get_type(self) -> SimileType:
        """Returns the resulting type of the operation/expression/statement represented by this AST node.

        Initially, :cls:`Identifier` nodes will return a :cls:`DeferToSymbolTable` type.
        After running :func:`src.mod.analysis.type_analysis.populate_ast_with_types`, all nodes will contain resolved types.
        """
        raise NotImplementedError

    def contains(self, node: type[ASTNode], with_op_type: Operators | None = None) -> bool:
        """Check if the AST node contains a specific type of node."""

        def is_matching_node(n: Any) -> bool:
            if not isinstance(n, node):
                return False
            if with_op_type is None:
                return True
            if not hasattr(n, "op_type"):
                return False
            assert hasattr(n, "op_type")
            return n.op_type == with_op_type

        return any(dataclass_traverse(self, is_matching_node))

    def find_all_instances(self, type_: type[T], with_op_type: Operators | None = None) -> list[T]:
        """Returns a flattened list of all instances of a specific type in the AST.

        Most useful for finding identifiers nested within expressions.
        """

        def isinstance_with_op_type(n: Any) -> T | None:
            if not isinstance(n, type_):
                return None
            if with_op_type is None:
                return n
            if hasattr(n, "op_type") and n.op_type == with_op_type:
                return n
            return None

        return list(filter(None, dataclass_traverse(self, isinstance_with_op_type)))

    def children(self) -> Generator[ASTNode, None, None]:
        """Returns a list of all children AST nodes (only 1 level deep). Includes op_type fields if they exist."""
        for f in fields(self):
            field_value = getattr(self, f.name)
            if isinstance(field_value, list):
                for item in field_value:
                    yield item
            else:
                yield field_value

    def is_leaf(self) -> bool:
        """Check if the AST node is a leaf node (i.e., has no dataclass/list of dataclass children)."""
        for f in fields(self):
            field_value = getattr(self, f.name)
            if isinstance(field_value, list):
                if any(isinstance(item, ASTNode) for item in field_value):
                    return False
            elif isinstance(field_value, ASTNode):
                return False
        return True

    def pretty_print(
        self,
        ignore_fields: list[str] | None = None,
        indent=2,
    ) -> str:
        """Pretty print the AST node with JSON-like indentation."""
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
    """Identifier for variables, functions, etc. in the AST."""

    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identifier):
            return False
        return self.name == other.name

    def __post_init__(self) -> None:
        self.processed_type: SimileType | None = None

    @property
    def free(self) -> set[Identifier]:
        return {self}

    @property
    def get_type(self) -> SimileType:
        if self.processed_type:
            return self.processed_type
        return DeferToSymbolTable(lookup_type=self.name)
