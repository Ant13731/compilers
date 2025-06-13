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

    # Idea: store the first sets and the corresponding functions (that would otherwise be "matched" when making decisions)
    # It may be nice to allow for nested first sets and then a lookup (using the idea of getting all leaves from a tree...)
    first_sets: ClassVar[dict[str, dict[TokenType, Callable]]] = {
        # "statement": {""},
    }

    @property
    def eof(self) -> bool:
        return self.peek().type_ == TokenType.EOF

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[self.current_index + offset]

    def advance(self) -> Token:
        """Advance to the next token."""
        if not self.eof:
            self.current_index += 1
        return self.peek(-1)

    def check(self, token_type: TokenType) -> bool:
        return not self.eof and self.peek() == token_type

    def match(self, token_types: list[TokenType]) -> bool:
        for token_type in token_types:
            if self.check(token_type):
                self.advance()
                return True
        return False

    # Parsing based (loosely) on the grammar in grammar.lark
    def start(self) -> ast_.Start:
        if not self.tokens:
            return ast_.Start(ast_.None_())
        return ast_.Start(self.statements())

    def statements(self) -> ast_.Statements:

        statements = []
        while not self.eof:
            statement = None
            match self.peek().type_:

                case _:
                    self.errors.append(ParseError(f"Unexpected token. Expected one of {first_statement}", self.peek()))

            statements.append(statement)
        return ast_.Statements(statements)


def parse(tokens: list[Token]) -> ast_.ASTNode:
    """Parse a list of tokens into an abstract syntax tree (AST)."""
    parser = Parser(tokens)
    return parser.start()
