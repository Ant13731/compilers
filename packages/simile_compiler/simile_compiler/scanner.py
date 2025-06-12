from dataclasses import dataclass, field
from enum import Enum, auto


class ScanException(Exception):
    pass


class TokenType(Enum):
    """Valid token types for the Simile language."""

    # Formatting
    EOF = auto()  # End of file
    INDENT = auto()
    DEDENT = auto()

    # Primitives
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()

    # Identifiers
    IDENTIFIER = auto()

    # Notation
    ASSIGN = auto()
    CDOT = auto()
    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    VBAR = auto()

    COMMENT = auto()

    # Keywords
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    FOR = auto()  # purposefully no while loop?

    STRUCT = auto()
    ENUM = auto()

    DEF = auto()
    RIGHTARROW = auto()
    LAMBDA = auto()

    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()

    # Brackets
    L_PAREN = auto()
    R_PAREN = auto()
    L_BRACKET = auto()
    R_BRACKET = auto()
    L_BRACE = auto()
    R_BRACE = auto()
    L_BRACE_BAR = auto()
    R_BRACE_BAR = auto()

    # Operators
    EQUALS = auto()
    NOT_EQUALS = auto()
    IS = auto()
    IS_NOT = auto()

    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    BANG = auto()
    IMPLIES = auto()
    REV_IMPLIES = auto()
    EQUIVALENT = auto()
    NOT_EQUIVALENT = auto()

    FORALL = auto()
    EXISTS = auto()

    # Numbers
    PLUS = auto()
    MINUS = auto()
    STAR = auto()  # multiply
    SLASH = auto()  # divide
    PERCENT = auto()  # mod
    DOUBLE_STAR = auto()  # exponentiation

    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    # Sets
    ELEMENT_OF = auto()  # also used for in
    NOT_ELEMENT_OF = auto()  # also used for not in
    UNION = auto()
    INTERSECTION = auto()
    BACKSLASH = auto()  # set difference

    CARTESIAN_PRODUCT = auto()
    POWERSET = auto()
    NONEMPTY_POWERSET = auto()
    TILDE = auto()  # set complement

    SUBSET = auto()
    SUBSET_EQ = auto()
    SUPERSET = auto()
    SUPERSET_EQ = auto()

    NOT_SUBSET = auto()
    NOT_SUPERSET = auto()
    NOT_SUBSET_EQ = auto()
    NOT_SUPERSET_EQ = auto()

    UNION_ALL = auto()
    INTERSECTION_ALL = auto()

    # Relations
    MAPLET = auto()

    RELATION_OVERRIDING = auto()
    COMPOSITION = auto()
    INVERSE = auto()

    DOMAIN_SUBTRACTION = auto()
    DOMAIN_RESTRICTION = auto()
    RANGE_SUBTRACTION = auto()
    RANGE_RESTRICTION = auto()

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

    # direct/parallel product??

    # Sequences
    UPTO = auto()

    # Common sets
    NATURAL_NUMBERS = auto()
    POSITIVE_INTEGERS = auto()
    INTEGERS = auto()


OPERATOR_TOKEN_TABLE = {
    ":=": TokenType.ASSIGN,
    "·": TokenType.CDOT,
    ".": TokenType.DOT,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
    "|": TokenType.VBAR,
    "λ": TokenType.LAMBDA,
    "->": TokenType.RIGHTARROW,
    "(": TokenType.L_PAREN,
    ")": TokenType.R_PAREN,
    "[": TokenType.L_BRACKET,
    "]": TokenType.R_BRACKET,
    "{": TokenType.L_BRACE,
    "}": TokenType.R_BRACE,
    "⦃": TokenType.L_BRACE_BAR,
    "⦄": TokenType.R_BRACE_BAR,
    "{|": TokenType.L_BRACE_BAR,
    "|}": TokenType.R_BRACE_BAR,
    "=": TokenType.EQUALS,
    "≠": TokenType.NOT_EQUALS,
    "!=": TokenType.NOT_EQUALS,
    "!": TokenType.BANG,
    "∧": TokenType.AND,
    "∨": TokenType.OR,
    "¬": TokenType.NOT,
    "⇒": TokenType.IMPLIES,
    "⇐": TokenType.REV_IMPLIES,
    "⇔": TokenType.EQUIVALENT,
    "≡": TokenType.EQUIVALENT,
    "≢": TokenType.NOT_EQUIVALENT,
    "∀": TokenType.FORALL,
    "∃": TokenType.EXISTS,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "**": TokenType.DOUBLE_STAR,
    "<": TokenType.LT,
    "<=": TokenType.LE,
    ">": TokenType.GT,
    ">=": TokenType.GE,
    "∈": TokenType.ELEMENT_OF,
    "∉": TokenType.NOT_ELEMENT_OF,
    "∪": TokenType.UNION,
    "\\/": TokenType.UNION,
    "∩": TokenType.INTERSECTION,
    "/\\": TokenType.INTERSECTION,
    "∖": TokenType.BACKSLASH,
    "\\": TokenType.BACKSLASH,
    "×": TokenType.CARTESIAN_PRODUCT,
    "><": TokenType.CARTESIAN_PRODUCT,
    "~": TokenType.TILDE,
    "⊆": TokenType.SUBSET_EQ,
    "<:": TokenType.SUBSET_EQ,
    "⊂": TokenType.SUBSET,
    "<<:": TokenType.SUBSET,
    "⊇": TokenType.SUPERSET_EQ,
    ":>": TokenType.SUPERSET_EQ,
    "⊃": TokenType.SUPERSET,
    ":>>": TokenType.SUPERSET,
    "⊈": TokenType.NOT_SUBSET_EQ,
    "⊄": TokenType.NOT_SUBSET,
    "⊉": TokenType.NOT_SUPERSET_EQ,
    "⊅": TokenType.NOT_SUPERSET,
    "/<:": TokenType.SUBSET_EQ,
    "/<<:": TokenType.SUBSET,
    "/:>": TokenType.SUPERSET_EQ,
    "/:>>": TokenType.SUPERSET,
    "ℙ": TokenType.POWERSET,
    "ℙ₁": TokenType.NONEMPTY_POWERSET,
    "⋂": TokenType.INTERSECTION_ALL,
    "⋃": TokenType.UNION_ALL,
    "↦": TokenType.MAPLET,
    "|->": TokenType.MAPLET,
    "<+": TokenType.RELATION_OVERRIDING,  # TODO unicode version
    "∘": TokenType.COMPOSITION,
    "⁻¹": TokenType.INVERSE,
    "◁": TokenType.DOMAIN_RESTRICTION,
    "<|": TokenType.DOMAIN_RESTRICTION,
    "⩤": TokenType.DOMAIN_SUBTRACTION,
    "<<|": TokenType.DOMAIN_SUBTRACTION,
    "▷": TokenType.RANGE_RESTRICTION,
    "|>": TokenType.RANGE_RESTRICTION,
    "⩥": TokenType.RANGE_SUBTRACTION,
    "|>>": TokenType.RANGE_SUBTRACTION,
    "↔": TokenType.RELATION,
    "<->": TokenType.RELATION,
    "<<->": TokenType.TOTAL_RELATION,  # TODO unicode version
    "<->>": TokenType.SURJECTIVE_RELATION,  # TODO unicode version
    "<<->>": TokenType.TOTAL_SURJECTIVE_RELATION,  # TODO unicode version
    "⇸": TokenType.PARTIAL_FUNCTION,
    "+->": TokenType.PARTIAL_FUNCTION,
    "→": TokenType.TOTAL_FUNCTION,
    "-->": TokenType.TOTAL_FUNCTION,
    "⤔": TokenType.PARTIAL_INJECTION,
    ">+>": TokenType.PARTIAL_INJECTION,
    "↣": TokenType.TOTAL_INJECTION,
    ">->": TokenType.TOTAL_INJECTION,
    "⤀": TokenType.PARTIAL_SURJECTION,
    "+->>": TokenType.PARTIAL_SURJECTION,
    "↠": TokenType.TOTAL_SURJECTION,
    "-->>": TokenType.TOTAL_SURJECTION,
    "⤖": TokenType.BIJECTION,
    ">->>": TokenType.BIJECTION,
    "..": TokenType.UPTO,
    "ℤ": TokenType.INTEGERS,
    "ℕ": TokenType.NATURAL_NUMBERS,
    "ℕ₁": TokenType.POSITIVE_INTEGERS,
}

KEYWORD_TABLE = {
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "None": TokenType.NONE,
    "if": TokenType.IF,
    "elif": TokenType.ELIF,
    "else": TokenType.ELSE,
    "for": TokenType.FOR,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "def": TokenType.DEF,
    "is": TokenType.IS,
    "is not": TokenType.IS_NOT,
    "in": TokenType.ELEMENT_OF,
    "not in": TokenType.NOT_ELEMENT_OF,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "forall": TokenType.FORALL,
    "exists": TokenType.EXISTS,
    "powerset": TokenType.POWERSET,
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "circ": TokenType.COMPOSITION,
}


@dataclass
class Location:
    """Represents the location of a token in the source code."""

    line: int
    column: int


@dataclass
class Token:
    """Represents a single token in the Simile language."""

    type_: TokenType
    value: str
    start_location: Location
    end_location: Location

    def __repr__(self) -> str:
        if self.value:
            return f"{self.type_.name}({self.value})"
        return self.type_.name


@dataclass
class Scanner:
    """Configuration options for the scanner."""

    text: str

    current_index_lexeme_start: int = 0
    current_index: int = 0
    location: Location = field(default_factory=lambda: Location(0, 0))

    # Since we want significant whitespace, we need to preserve indentation context
    indentation_stack: list[str] = field(default_factory=list)

    scanned_tokens: list[Token] = field(default_factory=list)

    @property
    def max_location(self) -> Location:
        """Returns the maximum location in the text."""
        return Location(len(self.text.splitlines()), len(self.text.splitlines()[-1]) if self.text else 0)

    @property
    def at_end_of_text(self) -> bool:
        """Checks if the scanner has reached the end of the text."""
        return self.current_index >= len(self.text)

    @property
    def current_lexeme_len(self) -> int:
        """Returns the length of the current lexeme being scanned."""
        return self.current_index - self.current_index_lexeme_start

    def peek(self, offset: int = 0) -> str | None:
        if self.at_end_of_text:
            return None
        return self.text[self.current_index + offset]

    def advance(self) -> str:
        c = self.text[self.current_index]
        self.current_index += 1
        self.location.column += 1
        return c

    def match(self, expected: str) -> bool:
        assert len(expected) == 1  # only one character of matching
        if self.at_end_of_text:
            return False
        if self.peek() != expected:
            return False
        self.advance()
        return True

    def match_phrase(self, expected: str) -> bool:
        expected_index = 0
        while expected_index < len(expected) and self.match(expected[expected_index]):
            expected_index += 1
        return expected_index == len(expected)

    def add_token(self, type_: TokenType, start_location: Location | None = None, end_location: Location | None = None, value: str = "") -> None:
        """Adds a token to the scanner's output.

        Args:
            type_ (TokenType): The type of the token.
            start_location (Location, optional): The starting location of the token. Defaults to current location.
            end_location (Location, optional): The ending location of the token. Defaults to current location.
            value (str, optional): The value of the token. Defaults to an empty string.
        """
        if start_location is None:
            start_location = self.location
        if end_location is None:
            end_location = self.location

        self.scanned_tokens.append(Token(type_, value, start_location, end_location))

    def scan_next(self) -> None:
        # End of file check
        if self.peek() is None:
            self.add_token(TokenType.EOF)
            return

        c = self.advance()

        # Skip newlines
        if c == "\n":
            self.location.line += 1
            self.location.column = 0
            return

        # Handle indentation
        # If we are at the start of a line...
        if self.location.column == 1:
            # Match existing indentation as much as possible
            matched_up_to_index = 0
            for indent_str in self.indentation_stack:
                if not self.match_phrase(indent_str):
                    break
                matched_up_to_index += 1

            # Indentation doesn't match but no content on line - ignore indentation for line
            if self.peek() == "\n" or self.peek() == "#":
                return

            indentation_difference = matched_up_to_index - len(self.indentation_stack)

            if indentation_difference < 0:
                if self.peek() == " " or self.peek() == "\t":
                    # There is more "indentation" left to consume, but the leftover does not match what we expect so far.
                    raise ScanException(
                        self.location,
                        self.peek(),
                        f"Indentation does not match. Expected {self.indentation_stack[indentation_difference:]} but got {self.move_until_no_whitespace()}",
                    )
                # Next character is another token with less indentation, so record all required dedents
                self.indentation_stack = self.indentation_stack[:indentation_difference]
                for _ in range(-indentation_difference):
                    self.add_token(TokenType.DEDENT)
            else:  # we've matched up to the indentation point
                if self.peek() == " " or self.peek() == "\t":
                    # Add new indentation level
                    new_indent = self.move_until_no_whitespace()
                    # ignore new indentation on blank/comment lines
                    if self.peek() == "\n" or self.peek() == "#":
                        return
                    self.indentation_stack.append(new_indent)
                    self.add_token(TokenType.INDENT)
                # Maintain indentation level
                return

        # Need to handle dedents back to the first level
        # if self.location.column == 1 and c not in " \t\n":
        #     # All indentations will need to be matched with a corresponding dedent (these are basically like parenthesis)
        #     for indentation_level in self.indentation_stack:
        #         self.add_token(TokenType.DEDENT)
        match c:
            case "\r":
                return
            case " " | "\t":
                # self.location.column will never be equal to 1 here (covered above)
                return

                # We must be at the start of a line (and not in a set, sequence, or relation)
                # if self.location.column != 1:  # also ignore when we are inside a set, sequence, or relation?
                #     return

                # # Indentation should always match up until the topmost indentation level
                # for indent_str in self.indentation_stack[:-1]:
                #     if self.match_phrase(indent_str):
                #         continue
                #     # Indentation doesn't match but no content on line - ignore line
                #     if self.peek() == "\n":
                #         return
                #     # Indentation doesn't match
                #     # if not self.indentation_stack or indentation_level_difference != -self.indentation_stack[-1]:
                #     raise ScanException(self.location, self.peek(), f"Indentation at {self.location} does not match previous indentations")

                # if not self.indentation_stack or self.match_phrase(self.indentation_stack[-1]):
                #     peek = self.peek()
                #     if peek is None or peek not in " \t":
                #         return

                #     # We are entering a new indentation level
                #     new_indentation = self.move_until_no_whitespace()
                #     # (ignoring empty lines, of course)
                #     if self.peek() == "\n" or self.at_end_of_text:
                #         return

                #     self.indentation_stack.append(new_indentation)
                #     self.add_token(TokenType.INDENT)
                #     return
                # else:
                #     if self.peek() == "\n":
                #         return
                #     # We are leaving an indentation level
                #     self.indentation_stack.pop()
                #     self.add_token(TokenType.DEDENT)
                #     return
            # # Notation
            # case ":":
            #     if self.match("="):
            #         self.add_token(TokenType.ASSIGN)
            #     else:
            #         self.add_token(TokenType.COLON)
            # case "·":
            #     self.add_token(TokenType.DOT)
            # case ".":
            #     if self.match("."):
            #         self.add_token(TokenType.UPTO)
            #     else:
            #         self.add_token(TokenType.DOT)
            # case ",":
            #     self.add_token(TokenType.COMMA)
            # case ";":
            #     self.add_token(TokenType.SEMICOLON)
            # case "|":
            #     if self.match("}"):
            #         self.add_token(TokenType.R_BRACE_BAR)
            #     elif self.match("-"):
            #         if self.match(">"):
            #             self.add_token(TokenType.MAPLET)
            #         else:
            #             raise ScanException(self.location, self.peek(), "Unexpected character after '|-' (expected '>')")
            #     elif self.match(">"):
            #         if self.match("-"):
            #             self.add_token(TokenType.RANGE_SUBTRACTION)
            #         else:
            #             self.add_token(TokenType.RANGE_RESTRICTION)
            #     else:
            #         self.add_token(TokenType.VBAR)
            # # Functions
            # case "λ":
            #     self.add_token(TokenType.LAMBDA)
            # case "→":
            #     self.add_token(TokenType.RIGHTARROW)
            # # Brackets
            # case "(":
            #     self.add_token(TokenType.L_PAREN)
            # case ")":
            #     self.add_token(TokenType.R_PAREN)
            # case "[":
            #     self.add_token(TokenType.L_BRACKET)
            # case "]":
            #     self.add_token(TokenType.R_BRACKET)
            # case "{":
            #     if self.match("|"):
            #         self.add_token(TokenType.L_BRACE_BAR)
            #     else:
            #         self.add_token(TokenType.L_BRACE)
            # case "}":
            #     self.add_token(TokenType.R_BRACE)
            # # Operators
            # case "=":
            #     self.add_token(TokenType.EQUALS)
            #     if self.match("="):
            #         if self.match(">"):
            #             self.add_token(TokenType.IMPLIES)
            # case "≠":
            #     self.add_token(TokenType.NOT_EQUALS)
            # case "!":
            #     if self.match("="):
            #         self.add_token(TokenType.NOT_EQUALS)
            #     elif self.match("≡"):
            #         self.add_token(TokenType.EQUIVALENT)
            #     else:
            #         self.add_token(TokenType.BANG)
            # # Logical
            # case "∧":
            #     self.add_token(TokenType.AND)
            # case "∨":
            #     self.add_token(TokenType.OR)
            # case "¬":
            #     self.add_token(TokenType.NOT)
            # case "⇒":
            #     self.add_token(TokenType.IMPLIES)
            # case "⇐":
            #     self.add_token(TokenType.REV_IMPLIES)
            # case "⇔":
            #     self.add_token(TokenType.EQUIVALENT)
            # case "≡":
            #     self.add_token(TokenType.EQUIVALENT)
            # case "≢":
            #     self.add_token(TokenType.NOT_EQUIVALENT)
            # case "∀":
            #     self.add_token(TokenType.FORALL)
            # case "∃":
            #     self.add_token(TokenType.EXISTS)
            # # Numerical operators
            # case "+":
            #     if self.match("-"):
            #         if self.match(">"):
            #             self.add_token(TokenType.PARTIAL_FUNCTION)
            #         else:
            #             raise ScanException(self.location, self.peek(), "Unexpected character after '+-' (expected '>')")
            #     else:
            #         self.add_token(TokenType.PLUS)
            # case "-":
            #     if self.match("-"):
            #         if self.match(">"):
            #             self.add_token(TokenType.PARTIAL_FUNCTION)
            #         else:
            #             raise ScanException(self.location, self.peek(), "Unexpected character after '--' (expected '>')")
            #     else:
            #         if self.match(">"):
            #             self.add_token(TokenType.RIGHTARROW)
            #         else:
            #             self.add_token(TokenType.MINUS)
            # case "*":
            #     if self.match("*"):
            #         self.add_token(TokenType.DOUBLE_STAR)
            #     else:
            #         self.add_token(TokenType.STAR)
            # case "/":
            #     if self.match("\\"):
            #         self.add_token(TokenType.INTERSECTION)
            #     else:
            #         self.add_token(TokenType.SLASH)
            # case "%":
            #     self.add_token(TokenType.PERCENT)
            # case "<":
            #     match self.advance():
            #         case "=":
            #             if self.match("="):
            #                 if self.match(">"):
            #                     self.add_token(TokenType.EQUIVALENT)
            #                 else:
            #                     self.add_token(TokenType.REV_IMPLIES)
            #             else:
            #                 self.add_token(TokenType.LE)

            #         case "!":
            #             if self.match("="):
            #                 if self.match("="):
            #                     if self.match(">"):
            #                         self.add_token(TokenType.NOT_EQUIVALENT)
            #                     else:
            #                         raise ScanException(self.location, self.peek(), "Unexpected character after '<!==' (expected '>')")
            #                 else:
            #                     raise ScanException(self.location, self.peek(), "Unexpected character after '<!=' (expected '=')")
            #             else:
            #                 raise ScanException(self.location, self.peek(), "Unexpected character after '<!' (expected '=')")
            #         case "|":
            #             match c2 := self.advance():
            #                 case "-":
            #                     self.add_token(TokenType.RELATION_OVERRIDING)
            #             self.add_token(TokenType.DOMAIN_RESTRICTION)
            #         case "<":
            #             match c2 := self.advance():
            #                 case "|":
            #                     self.add_token(TokenType.DOMAIN_SUBTRACTION)
            #         case _:
            #             self.add_token(TokenType.LT)

            # case ">":
            #     if self.match("="):
            #         self.add_token(TokenType.GE)
            #     elif self.match("<"):
            #         self.add_token(TokenType.CARTESIAN_PRODUCT)
            #     else:
            #         self.add_token(TokenType.GT)
            # # Sets
            # case "∈":
            #     self.add_token(TokenType.ELEMENT_OF)
            # case "∉":
            #     self.add_token(TokenType.NOT_ELEMENT_OF)
            # case "∪":
            #     self.add_token(TokenType.UNION)
            # case "∩":
            #     self.add_token(TokenType.INTERSECTION)
            # case "∖":
            #     self.add_token(TokenType.BACKSLASH)
            # case "\\":
            #     if self.match("/"):
            #         self.add_token(TokenType.UNION)
            #     else:
            #         self.add_token(TokenType.BACKSLASH)
            # case "×":
            #     self.add_token(TokenType.CARTESIAN_PRODUCT)
            # case "~":
            #     self.add_token(TokenType.TILDE)
            # case "⊆":
            #     self.add_token(TokenType.SUBSET_EQ)
            # case "⊂":
            #     self.add_token(TokenType.SUBSET)
            # case "⊇":
            #     self.add_token(TokenType.SUPERSET_EQ)
            # case "⊃":
            #     self.add_token(TokenType.SUPERSET)
            # case "⊈":
            #     self.add_token(TokenType.NOT_SUBSET_EQ)
            # case "⊄":
            #     self.add_token(TokenType.NOT_SUBSET)
            # case "⊉":
            #     self.add_token(TokenType.NOT_SUPERSET_EQ)
            # case "⊅":
            #     self.add_token(TokenType.NOT_SUPERSET)
            # case "ℙ":
            #     if self.match("₁"):
            #         self.add_token(TokenType.NONEMPTY_POWERSET)
            #     else:
            #         self.add_token(TokenType.POWERSET)
            # case "⋂":
            #     self.add_token(TokenType.INTERSECTION_ALL)
            # case "⋃":
            #     self.add_token(TokenType.UNION_ALL)
            # # Relations
            # case "↦":
            #     self.add_token(TokenType.MAPLET)
            # case "∘":
            #     self.add_token(TokenType.COMPOSITION)
            # # TODO relation overriding operator (have ascii version already)
            # case "⁻":
            #     if self.match("¹"):
            #         self.add_token(TokenType.INVERSE)
            #     else:
            #         raise ScanException(self.location, self.peek(), "Unexpected character after '⁻' (expected '¹')")
            # case "↔":
            #     self.add_token(TokenType.RELATION)
            # # TODO total relation, surjective relation, total surjective relation
            # case "⇸":
            #     self.add_token(TokenType.PARTIAL_FUNCTION)
            # case "⤔":
            #     self.add_token(TokenType.PARTIAL_INJECTION)
            # case "↣":
            #     self.add_token(TokenType.TOTAL_INJECTION)
            # case "⤀":
            #     self.add_token(TokenType.PARTIAL_SURJECTION)
            # case "↠":
            #     self.add_token(TokenType.TOTAL_SURJECTION)
            # case "⤖":
            #     self.add_token(TokenType.BIJECTION)

            # Comment
            case "#":
                start_location = self.location
                while not self.at_end_of_text and self.peek() != "\n":
                    self.advance()
                end_location = self.location
                value = self.text[self.current_index_lexeme_start + 1 : self.current_index]
                self.add_token(TokenType.COMMENT, start_location, end_location, value)
            # Primitives
            case '"':  # includes multiline?
                start_location = self.location
                while self.peek() != '"' and not self.at_end_of_text:
                    if self.peek() == "\n":
                        self.location.line += 1
                        self.location.column = 0
                    self.advance()

                if self.at_end_of_text:
                    raise ScanException(self.location, self.peek(), f'Unterminated string literal (expected ", found {self.peek()})')
                self.advance()  # will match \"
                value = self.text[self.current_index_lexeme_start + 1 : self.current_index - 1]
                self.add_token(TokenType.STRING, start_location, self.location, value)
            case _ if c.isdigit() or c == "." and (peek := self.peek()) is not None and peek.isdigit():
                start_location = self.location
                while (peek := self.peek()) is not None and peek.isdigit():
                    self.advance()
                if self.peek() == "." and (peek := self.peek(1)) is not None and (peek.isdigit() or peek.isspace()):
                    self.advance()
                while (peek := self.peek()) is not None and peek.isdigit():
                    self.advance()
                value = self.text[self.current_index_lexeme_start : self.current_index]
                if "." in value:
                    self.add_token(TokenType.FLOAT, start_location, self.location, value)
                else:
                    self.add_token(TokenType.INTEGER, start_location, self.location, value)
            # Operators
            case _ if c in {k[0] for k in OPERATOR_TOKEN_TABLE}:
                start_location = self.location

                consumed_characters = c
                possible_tokens: dict[str, TokenType] = {k: v for (k, v) in OPERATOR_TOKEN_TABLE.items() if k.startswith(consumed_characters)}
                # prev_possible_tokens = OPERATOR_TOKEN_TABLE

                # Eliminate possible tokens until the dictionary is empty
                while True:
                    peek = self.peek()
                    if peek is None:
                        break
                    peek_possible_tokens = {k: v for k, v in OPERATOR_TOKEN_TABLE.items() if len(k) > len(consumed_characters) and k.startswith(consumed_characters + peek)}
                    if len(peek_possible_tokens) == 0:
                        break
                    consumed_characters += self.advance()
                    possible_tokens = {k: v for (k, v) in OPERATOR_TOKEN_TABLE.items() if k.startswith(consumed_characters)}

                # while len(possible_tokens) > 0:
                #     prev_possible_tokens = possible_tokens
                #     peek = self.peek()
                #     if peek is None:
                #         break
                #     possible_tokens = {k: v for k, v in OPERATOR_TOKEN_TABLE.items() if len(k) > len(consumed_characters) and k.startswith(consumed_characters + peek)}
                #     consumed_characters += self.advance()

                # If token is valid, the previous iteration of possible tokens should contain the exact token we are looking for
                # if len(possible_tokens) < 1:
                #     raise ScanException(
                #         self.location,
                #         self.peek(),
                #         f"Symbol {consumed_characters} has multiple possible matches in the table {possible_tokens}, but none were valid with the next character {self.peek()}",
                #     )
                if possible_tokens.get(consumed_characters) is None:
                    raise ScanException(
                        self.location,
                        self.peek(),
                        f"Cannot find symbol {consumed_characters} in operator token table. Possible matches are {possible_tokens}, but none were valid with the next character {self.peek()}",
                    )

                self.add_token(OPERATOR_TOKEN_TABLE[consumed_characters], start_location, self.location)
            case _ if c.isalpha() or c == "_":
                start_location = self.location
                while True:
                    c_ = self.peek()
                    if c_ is None:
                        break
                    if not (c_.isalnum() or c_ == "_"):
                        break
                    self.advance()
                value = self.text[self.current_index_lexeme_start : self.current_index]
                token_type = KEYWORD_TABLE.get(value, TokenType.IDENTIFIER)
                self.add_token(token_type, start_location, self.location, value)
            case _:
                raise ScanException(self.location, self.peek(), "Unexpected character")

    def move_to_next_whitespace(self) -> None:
        """Advances the scanner past the next whitespace character in the text."""
        while not self.at_end_of_text:
            char = self.advance()
            if char.isspace():
                return None

    def move_until_no_whitespace(self) -> str:
        """Advances the scanner until it reaches a non-whitespace character.

        NOTE: This does not include newlines (in which case we can just ignore the entire blank line)"""
        moved_characters = ""
        while (c := self.peek()) is not None and c.isspace() and c != "\n":
            moved_characters += self.advance()
        return moved_characters


# TODO good error handling. need to catch as many errors as possible and return them instead of tokens
def scan(text: str) -> list[Token]:
    """Converts Simile source code into a list of tokens.

    Args:
        text (str): Simile source code (may be multiline)

    Returns:
        list[Token]: A list of tokens extracted from the source code
    """

    scanner = Scanner(text)

    while not scanner.at_end_of_text:
        try:
            scanner.current_index_lexeme_start = scanner.current_index
            scanner.scan_next()
        except ScanException as e:
            print(f"Error at {scanner.location}: {e}")
            scanner.move_to_next_whitespace()

    return scanner.scanned_tokens
