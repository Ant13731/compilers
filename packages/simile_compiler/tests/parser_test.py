import pytest


from src.mod.parser import parse, Parser
from src.mod.scanner import scan, TokenType
from src.mod.ast_ import *


def start_prefix(ast: ASTNode) -> ASTNode:
    """Wraps the ASTNode in a Start node."""
    return Start(Statements([ast]))


manual_tests = dict(
    map(
        lambda item: (item[0], start_prefix(item[1])),
        {
            "a": Identifier("a"),
            "(a)": Identifier("a"),
            "1": Int("1"),
            "1.": Float("1."),
            # ".": Float("."),
            "1.1": Float("1.1"),
            ".1": Float(".1"),
            '"Test"': String("Test"),
            '"\\""': String('\\"'),
            "True": True_(),
            "False": False_(),
            "None": None_(),
            "{}": Enumeration([], op_type=CollectionOperator.SET),
            "{ }": Enumeration([], op_type=CollectionOperator.SET),
            "{| |}": Enumeration([], op_type=CollectionOperator.BAG),
            "{||}": Enumeration([], op_type=CollectionOperator.BAG),
            "[]": Enumeration([], op_type=CollectionOperator.SEQUENCE),
            "[ ]": Enumeration([], op_type=CollectionOperator.SEQUENCE),
            "x <-> y": RelationOp(Identifier("x"), Identifier("y"), op_type=RelationOperator.RELATION),
            "x + y + z": BinaryOp(BinaryOp(Identifier("x"), Identifier("y"), op_type=BinaryOperator.ADD), Identifier("z"), op_type=BinaryOperator.ADD),
            "x > y": BinaryOp(Identifier("x"), Identifier("y"), op_type=BinaryOperator.GREATER_THAN),
            "x ==> y": BinaryOp(Identifier("x"), Identifier("y"), op_type=BinaryOperator.IMPLIES),
            "x ==> y ==> z": BinaryOp(BinaryOp(Identifier("x"), Identifier("y"), op_type=BinaryOperator.IMPLIES), Identifier("z"), op_type=BinaryOperator.IMPLIES),
            "x <== y <== z": BinaryOp(Identifier("x"), BinaryOp(Identifier("y"), Identifier("z"), op_type=BinaryOperator.REV_IMPLIES), op_type=BinaryOperator.REV_IMPLIES),
            "x |-> y": BinaryOp(Identifier("x"), Identifier("y"), op_type=BinaryOperator.MAPLET),
            "{x | x in [1, 2, 3]}": Comprehension(
                IdentList([]),
                BinaryOp(
                    Identifier("x"),
                    Enumeration([Int("1"), Int("2"), Int("3")], op_type=CollectionOperator.SEQUENCE),
                    op_type=BinaryOperator.IN,
                ),
                Identifier("x"),
                op_type=CollectionOperator.SET,
            ),
            "{ x |-> y | x in [1, 2, 3] and y in [4, 5, 6] }": Comprehension(
                IdentList([]),
                ListOp(
                    [
                        BinaryOp(
                            Identifier("x"),
                            Enumeration([Int("1"), Int("2"), Int("3")], op_type=CollectionOperator.SEQUENCE),
                            op_type=BinaryOperator.IN,
                        ),
                        BinaryOp(
                            Identifier("y"),
                            Enumeration([Int("4"), Int("5"), Int("6")], op_type=CollectionOperator.SEQUENCE),
                            op_type=BinaryOperator.IN,
                        ),
                    ],
                    op_type=ListOperator.AND,
                ),
                BinaryOp(
                    Identifier("x"),
                    Identifier("y"),
                    op_type=BinaryOperator.MAPLET,
                ),
                op_type=CollectionOperator.RELATION,
            ),
            "{ x, y · x in [1, 2, 3] and y in [4, 5, 6] | x |-> y }": Comprehension(
                IdentList([Identifier("x"), Identifier("y")]),
                ListOp(
                    [
                        BinaryOp(
                            Identifier("x"),
                            Enumeration([Int("1"), Int("2"), Int("3")], op_type=CollectionOperator.SEQUENCE),
                            op_type=BinaryOperator.IN,
                        ),
                        BinaryOp(
                            Identifier("y"),
                            Enumeration([Int("4"), Int("5"), Int("6")], op_type=CollectionOperator.SEQUENCE),
                            op_type=BinaryOperator.IN,
                        ),
                    ],
                    op_type=ListOperator.AND,
                ),
                BinaryOp(
                    Identifier("x"),
                    Identifier("y"),
                    op_type=BinaryOperator.MAPLET,
                ),
                op_type=CollectionOperator.RELATION,
            ),
            """
struct A:
    a: int,
    b: str, d: int
    c: float
""": StructDef(
                Identifier("A"),
                [
                    TypedName(Identifier("a"), Type_(Identifier("int"))),
                    TypedName(Identifier("b"), Type_(Identifier("str"))),
                    TypedName(Identifier("d"), Type_(Identifier("int"))),
                    TypedName(Identifier("c"), Type_(Identifier("float"))),
                ],
            ),
            f"B := {{A,B,C}}": Assignment(
                Identifier("B"),
                Enumeration(
                    [
                        Identifier("A"),
                        Identifier("B"),
                        Identifier("C"),
                    ],
                    op_type=CollectionOperator.SET,
                ),
            ),
            """
def test_func(a: int, b: str) -> bool:
    return a > 0 and b != ""
""": ProcedureDef(
                Identifier("test_func"),
                [
                    TypedName(Identifier("a"), Type_(Identifier("int"))),
                    TypedName(Identifier("b"), Type_(Identifier("str"))),
                ],
                Statements(
                    [
                        Return(
                            ListOp(
                                [
                                    BinaryOp(Identifier("a"), Int("0"), op_type=BinaryOperator.GREATER_THAN),
                                    BinaryOp(Identifier("b"), String(""), op_type=BinaryOperator.NOT_EQUAL),
                                ],
                                op_type=ListOperator.AND,
                            )
                        ),
                    ]
                ),
                Type_(Identifier("bool")),
            ),
            'import "test_import"': Import("test_import", None_()),
            'from "test_import" import *': Import("test_import", ImportAll()),
            'from "test_import" import test': Import("test_import", IdentList([Identifier("test")])),
            """
for i in [1, 2, 3]:
    print(i)
""": For(
                IdentList([Identifier("i")]),
                Enumeration(
                    [Int("1"), Int("2"), Int("3")],
                    op_type=CollectionOperator.SEQUENCE,
                ),
                Statements([Call(Identifier("print"), [Identifier("i")])]),
            ),
            """
while True:
    print("Hello")
""": While(
                condition=True_(),
                body=Statements([Call(Identifier("print"), [String("Hello")])]),
            ),
            """
if a > b:
    print("a is greater")
elif a < b:
    print("b is greater")
else:
    print("a and b are equal")
""": If(
                condition=BinaryOp(Identifier("a"), Identifier("b"), op_type=BinaryOperator.GREATER_THAN),
                body=Statements(
                    [
                        Call(Identifier("print"), [String("a is greater")]),
                    ]
                ),
                else_body=Elif(
                    condition=BinaryOp(Identifier("a"), Identifier("b"), op_type=BinaryOperator.LESS_THAN),
                    body=Statements(
                        [
                            Call(Identifier("print"), [String("b is greater")]),
                        ]
                    ),
                    else_body=Else(
                        body=Statements(
                            [
                                Call(Identifier("print"), [String("a and b are equal")]),
                            ]
                        )
                    ),
                ),
            ),
            "a := 1": Assignment(Identifier("a"), Int("1")),
            "a: int := 1": Assignment(TypedName(Identifier("a"), Type_(Identifier("int"))), Int("1")),
        }.items(),
    )
)
for k, v in manual_tests.items():
    print(f"Testing: {k}")
    print(scan(k))
    pk = parse(k)
    print(v)
    if not isinstance(pk, list):
        print(pk)
        print(pk.pretty_print())
        continue
    print("\nParse errors:")
    for err in pk:
        print()
        print(err)
    break


class TestParser:
    @pytest.mark.parametrize("rule", Parser.first_sets)
    def test_no_inf_loop_first_set(self, rule):
        print(f"Testing first set for rule: {rule}")
        first_set = Parser.get_first_set(rule)
        print(f"First set for {rule}: {first_set}")
        for token in first_set:
            assert isinstance(token, TokenType)

    @pytest.mark.parametrize("input_, expected", manual_tests.items())
    def test_manual(self, input_: str, expected: str):
        assert parse(input_) == expected
