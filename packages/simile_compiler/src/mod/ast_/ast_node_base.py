from __future__ import annotations
from dataclasses import dataclass, fields, field
from typing import Generator, Any, TypeVar, Callable
from functools import wraps

from src.mod.scanner import Location
from src.mod.ast_.ast_node_operators import Operators
from src.mod.ast_.dataclass_helpers import dataclass_traverse, dataclass_find_and_replace
from src.mod.ast_.symbol_table_types import SimileType, DeferToSymbolTable, SimileTypeError, PairType
from src.mod.ast_.symbol_table_env import SymbolTableEnvironment

T = TypeVar("T")


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    def __post_init__(self) -> None:
        self._env: SymbolTableEnvironment | None = None
        self._start_location: Location | None = None
        self._end_location: Location | None = None
        self._file_location: str | None = None

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
        """Returns the type of the AST node.

        Initially, :cls:`Identifier` nodes will return a :cls:`DeferToSymbolTable` type.
        After running :func:`src.mod.analysis.type_analysis.populate_ast_with_types`, all nodes will contain resolved types.
        """
        if self._env is None:
            raise SimileTypeError("Type analysis must be run before calling the `get_type` function (self._env is None)", self)
        return self._get_type()

    def _get_type(self) -> SimileType:
        """"""
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
            return n.op_type == with_op_type  # type: ignore

        return any(dataclass_traverse(self, is_matching_node))

    def contains_item(self, item: ASTNode | Any) -> bool:
        """Check if the AST node contains a specific item."""
        return any(dataclass_traverse(self, lambda n: n == item))

    def find_all_instances(self, type_: type[T], with_op_type: Operators | None = None) -> list[T]:
        """Returns a flattened list of all instances of a specific type in the AST.

        Most useful for finding identifiers nested within expressions.
        """

        def isinstance_with_op_type(n: Any) -> T | None:
            if not isinstance(n, type_):
                return None
            if with_op_type is None:
                return n
            if hasattr(n, "op_type") and n.op_type == with_op_type:  # type: ignore
                return n
            return None

        return list(filter(None, dataclass_traverse(self, isinstance_with_op_type)))

    def children(self, ast_nodes_only: bool = False) -> Generator[ASTNode | Any, None, None]:
        """Returns a list of all children AST nodes (only 1 level deep). Includes op_type fields if they exist."""
        for f in fields(self):
            field_value = getattr(self, f.name)
            if isinstance(field_value, list):
                for item in field_value:
                    yield item
            else:
                if isinstance(field_value, ASTNode):
                    yield field_value
                elif not ast_nodes_only:
                    # If we are not filtering for ASTNodes only, yield the field value directly
                    yield field_value

    def find_and_replace(self, find: ASTNode | Any, replace: ASTNode | Any) -> ASTNode:
        """Find and replace AST nodes using a rewrite function.

        The rewrite function should return the new AST node or None if no replacement is needed.
        """

        def rewrite_func(node: ASTNode | Any) -> ASTNode | None:
            if node == find:
                return replace
            return None

        return dataclass_find_and_replace(self, rewrite_func)

    def find_and_replace_with_func(self, rewrite_func: Callable[[ASTNode | Any], ASTNode | None]) -> ASTNode:
        """Find and replace AST nodes using a rewrite function.

        The rewrite function should return the new AST node or None if no replacement is needed.
        """

        return dataclass_find_and_replace(self, rewrite_func)

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
        print_env: bool = False,
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

        if print_env:

            ret += f"{indent_}_env={self._env.table if self._env else self._env}\n"

        for f in fields(self):
            field_value = getattr(self, f.name)

            if isinstance(field_value, ASTNode):
                if len(fields(self)) == 1:
                    ret += f"{indent_}{field_value.pretty_print(ignore_fields, indent+2,print_env)}"
                else:
                    ret += f"{indent_}{f.name}={field_value.pretty_print(ignore_fields,indent + 2,print_env)}"

                continue

            if isinstance(field_value, list):
                ret += f"{indent_}[\n"
                for item in field_value:
                    if isinstance(item, ASTNode):
                        ret += f"{indent_}{item.pretty_print(ignore_fields,indent + 2,print_env)}"
                    else:
                        ret += f"{indent_}    {item}\n"
                ret += f"{indent_}]\n"
                continue

            ret += f"{indent_}{f.name}**: {field_value}\n"
        return ret

    def pretty_print_algorithmic(self, indent: int = 0) -> str:
        return self._pretty_print_algorithmic(indent)

    def _pretty_print_algorithmic(self, indent: int) -> str:
        raise NotImplementedError

    def add_location(self, start: Location, end: Location, file: str) -> None:
        self._start_location = start
        self._end_location = end
        self._file_location = file

    def get_location(self) -> str:
        return f"({self._file_location}:{self._start_location}:{self._end_location})"


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

    @property
    def free(self) -> set[Identifier]:
        return {self}

    def _get_type(self) -> SimileType:
        if self._env is not None:
            ret = self._env.get(self.name)
            if ret is not None:
                return ret
        return DeferToSymbolTable(lookup_type=self.name)

    def _pretty_print_algorithmic(self, indent: int) -> str:
        if self.name.startswith("*"):
            return f"x{''.join(filter(str.isnumeric, self.name))}"
        return self.name

    def flatten(self) -> set[Identifier]:
        """Used to simplify the flatten operation of :cls:`MapletIdentifier`"""
        return {self}


@dataclass
class MapletIdentifier(ASTNode):
    """Special variation of maplet used for binding loop and quantification variables (also hashable)"""

    left: MapletIdentifier | Identifier
    right: MapletIdentifier | Identifier

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MapletIdentifier):
            return False
        return self.left == other.left and self.right == other.right

    @property
    def free(self) -> set[Identifier]:
        return self.flatten()

    def _get_type(self) -> SimileType:
        return PairType(self.left.get_type, self.right.get_type)

    def _pretty_print_algorithmic(self, indent: int) -> str:
        return f"{self.left._pretty_print_algorithmic(indent)} ↦ {self.right._pretty_print_algorithmic(indent)}"

    def flatten(self) -> set[Identifier]:
        return self.left.flatten() | self.right.flatten()
