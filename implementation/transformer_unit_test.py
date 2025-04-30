import unittest

from parser import parse
from ast_ import *

unittest.util._MAX_LENGTH = 2000  # type: ignore


class TestParserRules(unittest.TestCase):
    """A sanity-check class for the plaintext->AST pipeline.

    The outline of this class generally follows a 1:1 mapping to the methods in the transformer.
    """

    def low_level_comparison(self, value: PrimitiveLiteral) -> Start:
        return self.wrap_stmt(self.wrap_expr(value))

    def wrap_stmt(self, value) -> Start:
        return Start(Statements([value]))

    def wrap_equivalence(self, value: PrimitiveLiteral) -> EquivalenceStmt:
        return EquivalenceStmt(value)

    def wrap_expr(self, value: PrimitiveLiteral) -> Expr:
        return Expr(self.wrap_equivalence(value))

    def test_int(self):
        test_str = "1"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Int(1)))

    def test_float(self):
        test_str = "1.0"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Float(1.0)))

    def test_string(self):
        test_str = '"test"'
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(String(test_str)))

    def test_none(self):
        test_str = "None"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(None_()))

    def test_bool(self):
        test_str = "True"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Bool(True)))

    def test_identifier(self):
        test_str = "a"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Identifier("a")))

    def test_power(self):
        test_str = "a ** b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Power(Identifier("a"), Identifier("b"))))

    def test_factor(self):
        test_str = "+a"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Pos(Identifier("a"))))

    def test_term(self):
        test_str = "a * b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Mul(Identifier("a"), Identifier("b"))))

    def test_num_and_set_expr(self):
        test_str = "a + b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Add(Identifier("a"), Identifier("b"))))

    def test_comparison(self):
        test_str = "a < b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Lt(Identifier("a"), Identifier("b"))))

    def test_negation(self):
        test_str = "not a"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Not(Identifier("a"))))

    def test_conjunction(self):
        test_str = "a and b and c"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(And([Identifier("a"), Identifier("b"), Identifier("c")])))

    def test_disjunction(self):
        test_str = "a or b or c"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Or([Identifier("a"), Identifier("b"), Identifier("c")])))

    def test_impl(self):
        test_str = "a ==> b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Implies(Identifier("a"), Identifier("b"))))

    def test_rev_impl(self):
        test_str = "a <== b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(RevImplies(Identifier("a"), Identifier("b"))))

    def test_equivalence(self):
        test_str = "a <==> b"
        result = parse(test_str)
        self.assertEqual(result, self.low_level_comparison(Equiv(EquivalenceStmt(Identifier("a")), Identifier("b"))))

    def test_call(self):
        test_str = "a(b, c)"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                self.wrap_expr(
                    Call(
                        Identifier("a"),
                        Arguments([self.wrap_expr(Identifier("b")), self.wrap_expr(Identifier("c"))]),
                    )
                ),
            ),
        )

    def test_indexing(self):
        test_str = "a[b]"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                self.wrap_expr(
                    Indexing(Identifier("a"), self.wrap_equivalence(Identifier("b"))),
                ),
            ),
        )

    def test_slice(self):
        test_str = "a[b:c]"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                self.wrap_expr(
                    Indexing(
                        Identifier("a"),
                        Slice(
                            [
                                self.wrap_equivalence(Identifier("b")),
                                self.wrap_equivalence(Identifier("c")),
                            ]
                        ),
                    ),
                ),
            ),
        )

    def test_assignment(self):
        test_str = "a: int = b"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                Assignment(
                    TypedName(Identifier("a"), Type_(self.wrap_equivalence(Identifier("int")))),
                    self.wrap_expr(Identifier("b")),
                )
            ),
        )

    def test_lambdef(self):
        test_str = "lambda (b: int, c: str): b + c"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                Expr(
                    LambdaDef(
                        ArgDef(
                            [
                                TypedName(Identifier("b"), Type_(self.wrap_equivalence(Identifier("int")))),
                                TypedName(Identifier("c"), Type_(self.wrap_equivalence(Identifier("str")))),
                            ]
                        ),
                        self.wrap_equivalence(Add(Identifier("b"), Identifier("c"))),
                    )
                ),
            ),
        )

    def test_return(self):
        test_str = "return a"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                Return(self.wrap_expr(Identifier("a"))),
            ),
        )

    def test_if(self):
        test_str = "if a: b"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                If(self.wrap_equivalence(Identifier("a")), self.wrap_expr(Identifier("b"))),
            ),
        )

    def test_if_else(self):
        test_str = "if a: b else: c"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                If(
                    self.wrap_equivalence(Identifier("a")),
                    self.wrap_expr(Identifier("b")),
                    Else(self.wrap_expr(Identifier("c"))),
                )
            ),
        )

    def test_for(self):
        test_str = "for i in a: b"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                For(IterableNames([Identifier("i")]), self.wrap_expr(Identifier("a")), self.wrap_expr(Identifier("b"))),
            ),
        )

    def test_struct(self):
        test_str = "struct a: \n\tb: int"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                Struct(
                    Identifier("a"),
                    [TypedName(Identifier("b"), Type_(self.wrap_equivalence(Identifier("int"))))],
                )
            ),
        )

    def test_enum(self):
        test_str = "enum a: \n\tb \n\tc"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                Enum(Identifier("a"), [Identifier("b"), Identifier("c")]),
            ),
        )

    def test_func(self):
        test_str = "def a(b: int, c: str) -> int: return b + c"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.wrap_stmt(
                Func(
                    Identifier("a"),
                    ArgDef(
                        [
                            TypedName(Identifier("b"), Type_(self.wrap_equivalence(Identifier("int")))),
                            TypedName(Identifier("c"), Type_(self.wrap_equivalence(Identifier("str")))),
                        ]
                    ),
                    Type_(self.wrap_equivalence(Identifier("int"))),
                    Return(self.wrap_expr(Add(Identifier("b"), Identifier("c")))),
                )
            ),
        )

    def test_set_comp(self):
        test_str = "{a | a in b}"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.low_level_comparison(
                SetComprehension(
                    self.wrap_expr(Identifier("a")),
                    self.wrap_expr(In(Identifier("a"), Identifier("b"))),
                )
            ),
        )

    def test_dict_comp(self):
        test_str = "{a: b | a in c}"
        result = parse(test_str)
        self.assertEqual(
            result,
            self.low_level_comparison(
                DictComprehension(
                    KeyPair(self.wrap_expr(Identifier("a")), self.wrap_expr(Identifier("b"))),
                    self.wrap_expr(In(Identifier("a"), Identifier("c"))),
                )
            ),
        )
