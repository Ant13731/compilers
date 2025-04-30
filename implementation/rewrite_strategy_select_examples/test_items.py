from ast_ import *
import copy


# Functions to help with manual AST construction
def expr_preamble(expr: BaseAST) -> Expr:
    """Wraps the expression in an EquivalenceStmt and Expr."""
    return Expr(EquivalenceStmt(expr))


def start_preamble(body: BaseAST) -> Start:
    """Wraps the body in a Start and Statements."""
    return Start(Statements([body]))


@dataclass
class ConcreteSetLoop(BaseAST):
    generator: Set
    conditions: In | And | Or | None_


@dataclass
class ConcreteSet(BaseAST):
    iterator_var: BaseAST
    loop_for: list[ConcreteSetLoop]


# Normally this would be part of the AST (through earlier assignments),
# but to keep the examples simple we will reuse some variables throughout the file.
# NOTE: I plan to remove the need for Expr(EquivalenceStmt(...)) wrappers in the future,
#   but for now these match with the current AST structure, unit tests, and grammar.
src_sets = {
    "S": Set(list(map(lambda x: expr_preamble(Int(x)), range(0, 5)))),
    "T": Set(list(map(lambda x: expr_preamble(Int(x)), range(3, 10)))),
    "literal": Set(list(map(lambda x: expr_preamble(Int(x)), [1, 2, 3]))),
}
symbol_lookup_table: dict[str, BinOp | PrimaryStmt] = {
    "S": SetComprehension(
        elem=expr_preamble(Identifier("x")),
        generator=expr_preamble(In(Identifier("x"), src_sets["S"])),
    ),
    "T": SetComprehension(
        elem=expr_preamble(Identifier("y")),
        generator=expr_preamble(In(Identifier("y"), src_sets["T"])),
    ),
    "A": Intersect(Identifier("S"), Identifier("T")),
}

# For now the set operators are written with backslashes, but this will certainly change in the future.
union = r"S \cup T"
intersection = r"S \cap T"
setminus = r"S \setminus T"
union_literal = r"S \cup {1, 2, 3}"
multiple_intersection = r"S \cap A"
nested_generators = r"{x | x in (S \cup T)}"
tests_1 = {
    union: start_preamble(expr_preamble(Union(symbol_lookup_table["S"], symbol_lookup_table["T"]))),
    intersection: start_preamble(expr_preamble(Intersect(symbol_lookup_table["S"], symbol_lookup_table["T"]))),
    setminus: start_preamble(expr_preamble(Difference(symbol_lookup_table["S"], symbol_lookup_table["T"]))),
    union_literal: start_preamble(
        expr_preamble(
            Union(
                symbol_lookup_table["S"],
                SetComprehension(
                    expr_preamble(Identifier("x")),
                    expr_preamble(In(Identifier("x"), src_sets["literal"])),
                ),
            ),
        )
    ),
    multiple_intersection: start_preamble(
        expr_preamble(
            Intersect(
                symbol_lookup_table["S"],
                Intersect(symbol_lookup_table["S"], symbol_lookup_table["T"]),
            )
        )
    ),
    nested_generators: start_preamble(
        expr_preamble(
            SetComprehension(
                elem=expr_preamble(Identifier("x")),
                generator=expr_preamble(
                    In(
                        Identifier("x"),
                        expr_preamble(Union(symbol_lookup_table["S"], symbol_lookup_table["T"])),
                    )
                ),
            )
        )
    ),
}
tests_1_denest = copy.deepcopy(tests_1)
tests_1_denest[nested_generators] = tests_1[union]
tests_2 = {
    union: start_preamble(
        expr_preamble(
            SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(Or([In(Identifier("x"), src_sets["S"]), In(Identifier("x"), src_sets["T"])])),
            )
        )
    ),
    intersection: start_preamble(
        expr_preamble(
            SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(And([In(Identifier("x"), src_sets["S"]), In(Identifier("x"), src_sets["T"])])),
            )
        )
    ),
    setminus: start_preamble(
        expr_preamble(
            SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(And([In(Identifier("x"), src_sets["S"]), Not(In(Identifier("x"), src_sets["T"]))])),
            )
        )
    ),
    union_literal: start_preamble(
        expr_preamble(
            SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(
                    Or(
                        [
                            In(Identifier("x"), src_sets["S"]),
                            In(Identifier("x"), src_sets["literal"]),
                        ]
                    ),
                ),
            )
        )
    ),
    multiple_intersection: start_preamble(
        expr_preamble(
            SetComprehension(
                expr_preamble(Identifier("x")),
                expr_preamble(
                    And(
                        [
                            In(Identifier("x"), src_sets["S"]),
                            In(Identifier("x"), src_sets["S"]),
                            In(Identifier("x"), src_sets["T"]),
                        ]
                    ),
                ),
            )
        )
    ),
    nested_generators: start_preamble(
        expr_preamble(
            SetComprehension(
                elem=expr_preamble(Identifier("x")),
                generator=expr_preamble(
                    Or([In(Identifier("x"), src_sets["S"]), In(Identifier("x"), src_sets["T"])]),
                ),
            )
        )
    ),
}

tests_3 = {
    union: start_preamble(
        expr_preamble(
            ConcreteSet(
                iterator_var=Identifier("x"),
                loop_for=[ConcreteSetLoop(src_sets["S"], None_()), ConcreteSetLoop(src_sets["T"], None_())],
            )
        )
    ),
    intersection: start_preamble(
        expr_preamble(
            ConcreteSet(
                iterator_var=Identifier("x"),
                loop_for=[ConcreteSetLoop(src_sets["S"], In(Identifier("x"), src_sets["T"]))],
            )
        )
    ),
    setminus: start_preamble(
        expr_preamble(
            ConcreteSet(
                iterator_var=Identifier("x"),
                loop_for=[ConcreteSetLoop(src_sets["S"], Not(In(Identifier("x"), src_sets["T"])))],
            )
        )
    ),
    union_literal: start_preamble(
        expr_preamble(
            ConcreteSet(
                iterator_var=Identifier("x"),
                loop_for=[ConcreteSetLoop(src_sets["S"], None_()), ConcreteSetLoop(src_sets["literal"], None_())],
            )
        )
    ),
    multiple_intersection: start_preamble(
        expr_preamble(
            ConcreteSet(
                iterator_var=Identifier("x"),
                loop_for=[ConcreteSetLoop(src_sets["S"], In(Identifier("x"), src_sets["T"]))],
            )
        )
    ),
    nested_generators: start_preamble(
        expr_preamble(
            ConcreteSet(
                iterator_var=Identifier("x"),
                loop_for=[ConcreteSetLoop(src_sets["S"], None_()), ConcreteSetLoop(src_sets["T"], None_())],
            )
        )
    ),
}
