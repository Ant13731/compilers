from enum import Enum, auto


class TokenType(Enum):
    """Valid token types for the Simile language."""

    # Formatting
    EOF = auto()  # End of file
    INDENT = auto()
    DEDENT = auto()
    NEWLINE = auto()

    # Imports
    FROM = auto()
    IMPORT = auto()

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
    WHILE = auto()

    STRUCT = auto()
    ENUM = auto()

    DEF = auto()
    RIGHTARROW = auto()
    LAMBDA = auto()

    RETURN = auto()
    BREAK = auto()
    CONTINUE = auto()
    PASS = auto()

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
    IN = auto()  # also used for in
    NOT_IN = auto()  # also used for not in
    UNION = auto()
    INTERSECTION = auto()
    BACKSLASH = auto()  # set difference

    CARTESIAN_PRODUCT = auto()
    POWERSET = auto()
    NONEMPTY_POWERSET = auto()
    CARDINALITY = auto()
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

    def __repr__(self) -> str:
        return f"TokenType.{self.name}"


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
    "==>": TokenType.IMPLIES,
    "⇐": TokenType.REV_IMPLIES,
    "<==": TokenType.REV_IMPLIES,
    "⇔": TokenType.EQUIVALENT,
    "<==>": TokenType.EQUIVALENT,
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
    "^": TokenType.DOUBLE_STAR,
    "<": TokenType.LT,
    "<=": TokenType.LE,
    ">": TokenType.GT,
    ">=": TokenType.GE,
    "∈": TokenType.IN,
    "∉": TokenType.NOT_IN,
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
    "": TokenType.TOTAL_RELATION,
    "<<->": TokenType.TOTAL_RELATION,  # TODO unicode version
    "": TokenType.SURJECTIVE_RELATION,
    "<->>": TokenType.SURJECTIVE_RELATION,  # TODO unicode version
    "": TokenType.TOTAL_SURJECTIVE_RELATION,
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
    "while": TokenType.WHILE,
    "struct": TokenType.STRUCT,
    "enum": TokenType.ENUM,
    "def": TokenType.DEF,
    "is": TokenType.IS,
    "is not": TokenType.IS_NOT,
    "in": TokenType.IN,
    "not in": TokenType.NOT_IN,
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
    "from": TokenType.FROM,
    "import": TokenType.IMPORT,
    "pass": TokenType.PASS,
    "lambda": TokenType.LAMBDA,
    "card": TokenType.CARDINALITY,
    # "fst": TokenType.FIRST,
    # "proj1": TokenType.FIRST,
    # "snd": TokenType.SECOND,
    # "proj2": TokenType.SECOND,
}
