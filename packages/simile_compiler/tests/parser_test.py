import pytest

from simile_compiler.parser import (  # type: ignore
    parse,
    Parser,
    ParseError,
)
from simile_compiler.scanner import scan, TokenType  # type: ignore
from simile_compiler.ast_ import *  # type: ignore # noqa: F401, F403


# test_strs = [
#     "a:=1\nb:=2",
#     "def test() -> int: return 42",
#     "def test() -> int: return \n def test() -> int: return 42",
#     "def test() -> int: return\n\n42",
# ]
# for str_ in test_strs:
#     print(f"Testing: {str_}")
#     print(parse(scan(str_)))
#     print(parse(scan(str_)).pretty_print())


def start_prefix(ast: ASTNode) -> ASTNode:
    """Wraps the ASTNode in a Start node."""
    return Start(Statements([ast]))


manual_tests = dict(
    map(
        lambda item: (item[0], start_prefix(item[1])),
        {
            # "a": Identifier("a"),
            # "(a)": Identifier("a"),
            # "1": Int("1"),
            # "1.": Float("1."),
            # # ".": Float("."),
            # "1.1": Float("1.1"),
            # ".1": Float(".1"),
            # '"Test"': String("Test"),
            # '"\\""': String('\\"'),
            # "True": True_(),
            # "False": False_(),
            # "None": None_(),
            # "{}": SetEnumeration([]),
            # "{ }": SetEnumeration([]),
            # "{| |}": BagEnumeration([]),
            # "{||}": BagEnumeration([]),
            # "[]": SequenceEnumeration([]),
            # "[ ]": SequenceEnumeration([]),
            "{x | x in [1, 2, 3]}": SetComprehension(
                IdentList([]),
                In(
                    Identifier("x"),
                    SequenceEnumeration(
                        [Int("1"), Int("2"), Int("3")],
                    ),
                ),
                Identifier("x"),
            ),
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
