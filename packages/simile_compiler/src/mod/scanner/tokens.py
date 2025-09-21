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
    NONDETERMINISTIC_MEMBERSHIP_ASSIGN = auto()
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
    WITH = auto()

    # Brackets
    L_PAREN = auto()
    R_PAREN = auto()
    L_BRACKET = auto()
    R_BRACKET = auto()
    L_BRACE = auto()
    R_BRACE = auto()
    L_DOUBLE_BRACKET = auto()
    R_DOUBLE_BRACKET = auto()

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
    MULT = auto()  # multiply
    DIV = auto()  # divide
    MOD = auto()  # mod
    EXPONENT = auto()  # exponentiation

    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    # Sets
    IN = auto()  # also used for in
    NOT_IN = auto()  # also used for not in
    UNION = auto()
    INTERSECTION = auto()
    SET_DIFFERENCE = auto()  # set difference

    CARTESIAN_PRODUCT = auto()
    POWERSET = auto()
    NONEMPTY_POWERSET = auto()
    # CARDINALITY = auto()
    # TILDE = auto()  # set complement

    SUBSET = auto()
    SUBSET_EQ = auto()
    SUPERSET = auto()
    SUPERSET_EQ = auto()

    NOT_SUBSET = auto()
    NOT_SUPERSET = auto()
    NOT_SUBSET_EQ = auto()
    NOT_SUPERSET_EQ = auto()

    GENERAL_UNION = auto()
    GENERAL_INTERSECTION = auto()

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
    # NATURAL_NUMBERS = auto()
    # POSITIVE_INTEGERS = auto()
    # INTEGERS = auto()

    def __repr__(self) -> str:
        return f"TokenType.{self.name}"

    def __str__(self) -> str:
        return self.name


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
    "⟦": TokenType.L_DOUBLE_BRACKET,
    "〚": TokenType.L_DOUBLE_BRACKET,
    "[[": TokenType.L_DOUBLE_BRACKET,
    "⟧": TokenType.R_DOUBLE_BRACKET,
    "〛": TokenType.R_DOUBLE_BRACKET,
    "]]": TokenType.R_DOUBLE_BRACKET,
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
    "<=!=>": TokenType.NOT_EQUIVALENT,
    "∀": TokenType.FORALL,
    "∃": TokenType.EXISTS,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.MULT,
    "÷": TokenType.DIV,
    "/": TokenType.DIV,
    "%": TokenType.MOD,
    "**": TokenType.EXPONENT,
    "^": TokenType.EXPONENT,
    "<": TokenType.LT,
    "≤": TokenType.LE,
    "<=": TokenType.LE,
    ">": TokenType.GT,
    "≥": TokenType.GE,
    ">=": TokenType.GE,
    "∈": TokenType.IN,
    "∉": TokenType.NOT_IN,
    "∪": TokenType.UNION,
    "\\/": TokenType.UNION,
    "∩": TokenType.INTERSECTION,
    "/\\": TokenType.INTERSECTION,
    "∖": TokenType.SET_DIFFERENCE,
    "\\": TokenType.SET_DIFFERENCE,
    "×": TokenType.CARTESIAN_PRODUCT,
    "><": TokenType.CARTESIAN_PRODUCT,
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
    "!<:": TokenType.NOT_SUBSET_EQ,
    "!<<:": TokenType.NOT_SUBSET,
    "!:>": TokenType.NOT_SUPERSET_EQ,
    "!:>>": TokenType.NOT_SUPERSET,
    "⋃": TokenType.GENERAL_UNION,
    "⋂": TokenType.GENERAL_INTERSECTION,
    "↦": TokenType.MAPLET,
    "|->": TokenType.MAPLET,
    "": TokenType.RELATION_OVERRIDING,
    "<+": TokenType.RELATION_OVERRIDING,  # TODO unicode version
    "∘": TokenType.COMPOSITION,
    "⁻¹": TokenType.INVERSE,
    "~": TokenType.INVERSE,
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
    "ℙ": TokenType.POWERSET,
    "ℙ₁": TokenType.NONEMPTY_POWERSET,
    "ℤ": TokenType.IDENTIFIER,
    "ℕ": TokenType.IDENTIFIER,
    "ℕ₁": TokenType.IDENTIFIER,
}

KEYWORD_TABLE = {
    # Primitive objects
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "None": TokenType.NONE,
    # Programming constructs
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
    "return": TokenType.RETURN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "circ": TokenType.COMPOSITION,
    "from": TokenType.FROM,
    "import": TokenType.IMPORT,
    "pass": TokenType.PASS,
    # Aliases for above symbols
    "forall": TokenType.FORALL,
    "exists": TokenType.EXISTS,
    "in": TokenType.IN,
    "not in": TokenType.NOT_IN,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
    "lambda": TokenType.LAMBDA,
    "union_all": TokenType.GENERAL_UNION,
    "intersection_all": TokenType.GENERAL_INTERSECTION,
    "powerset": TokenType.POWERSET,
    "powerset1": TokenType.NONEMPTY_POWERSET,
    # "card": TokenType.CARDINALITY,
    # "fst": TokenType.FIRST,
    # "proj1": TokenType.FIRST,
    # "snd": TokenType.SECOND,
    # "proj2": TokenType.SECOND,
}
