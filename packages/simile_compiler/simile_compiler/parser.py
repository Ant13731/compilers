from __future__ import annotations
from dataclasses import dataclass, fields, asdict, field, make_dataclass, is_dataclass
from typing import TypeVar, Generic, Callable, ClassVar, Any, TypeAlias

try:
    from .scanner import Token
except ImportError:
    from scanner import Token

try:
    from . import ast_  # type: ignore
except ImportError:
    import ast_  # type: ignore


@dataclass
class Parser:
    """Parser class to handle parsing of tokens into an AST."""

    tokens: list[Token]

    current_index: int

    # Parsing based (loosely) on the grammar in grammar.lark
    def start(self) -> ast_.Start:
        if not self.tokens:
            return ast_.Start(ast_.None_())


def parse(tokens: list[Token]) -> ast_.ASTNode:
    """Parse a list of tokens into an abstract syntax tree (AST)."""
    # TODO: Implement the parser
    pass
