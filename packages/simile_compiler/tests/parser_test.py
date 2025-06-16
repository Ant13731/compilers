import pytest

from simile_compiler.parser import (  # type: ignore
    parse,
    Parser,
    ParseError,
)
from simile_compiler.scanner import scan, TokenType  # type: ignore


test_strs = [
    "a:=1\nb:=2",
    "def test() -> int: return 42",
    "def test() -> int: return \n def test() -> int: return 42",
    "def test() -> int: return\n\n42",
]
for str_ in test_strs:
    print(f"Testing: {str_}")
    print(parse(scan(str_)))
    print(parse(scan(str_)).pretty_print())


class TestParser:
    @pytest.mark.parametrize("rule", Parser.first_sets)
    def no_inf_loop_first_set(self, rule):
        print(f"Testing first set for rule: {rule}")
        first_set = Parser.get_first_set(rule)
        print(f"First set for {rule}: {first_set}")
        for token in first_set:
            assert isinstance(token, TokenType)

    # def manual_test(self, input_: str, expected: str):
    #     parser = Parser(input_)
    #     assert parser.parse() == expected
