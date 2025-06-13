from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, ClassVar, Any, TypeAlias

try:
    from .scanner import Token, TokenType
except ImportError:
    from scanner import Token, TokenType  # type: ignore

try:
    from . import ast_
except ImportError:
    import ast_  # type: ignore


@dataclass
class ParseError:
    """Class to represent a parser error."""

    message: str
    token: Token | None = None

    def __str__(self) -> str:
        if self.token:
            return f"ParseError at {self.token.start_location}: {self.message}"
        return f"ParseError: {self.message}"


@dataclass
class Parser:
    """Parser class to handle parsing of tokens into an AST."""

    tokens: list[Token]
    current_index: int = 0
    errors: list[ParseError] = field(default_factory=list)

    @property
    def current_token(self) -> Token:
        return self.tokens[self.current_index]

    @property
    def eof(self) -> bool:
        return self.current_token.type_ == TokenType.EOF

    # Parsing based (loosely) on the grammar in grammar.lark
    def start(self) -> ast_.Start:
        if not self.tokens:
            return ast_.Start(ast_.None_())
        return ast_.Start(self.statements())

    def statements(self) -> ast_.Statements:
        statements = []
        while not self.eof:
            statement = self.statement()
            if statement:
                statements.append(statement)
        return ast_.Statements(statements)


def parse(tokens: list[Token]) -> ast_.ASTNode:
    """Parse a list of tokens into an abstract syntax tree (AST)."""
    parser = Parser(tokens)
    return parser.start()
