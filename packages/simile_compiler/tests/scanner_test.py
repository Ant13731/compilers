import pytest

from simile_compiler.scanner import (  # type: ignore
    scan,
    OPERATOR_TOKEN_TABLE,
    KEYWORD_TABLE,
    TokenType,
    # ScanException,
)

TOKENS_AND_KEYWORDS = list(OPERATOR_TOKEN_TABLE.items()) + list(KEYWORD_TABLE.items())
TOKENS_AND_KEYWORDS_NO_NOT = list(
    filter(
        lambda item: item[1]
        not in [
            TokenType.NOT,
            TokenType.NOT_IN,
            TokenType.IS_NOT,
        ],
        TOKENS_AND_KEYWORDS,
    )
)


class TestSymbols:
    @pytest.mark.parametrize(
        "input_1, expected_1",
        TOKENS_AND_KEYWORDS,
    )
    def test_single_symbol(self, input_1: str, expected_1: TokenType):
        assert list(map(lambda tk: tk.type_, scan(input_1))) == [expected_1, TokenType.EOF]

    @pytest.mark.parametrize(
        "input_1, expected_1",
        TOKENS_AND_KEYWORDS_NO_NOT,
    )
    @pytest.mark.parametrize(
        "input_2, expected_2",
        TOKENS_AND_KEYWORDS_NO_NOT,
    )
    def test_double_symbols_with_space(self, input_1: str, input_2: str, expected_1: TokenType, expected_2: TokenType):
        input_ = input_1 + " " + input_2
        assert list(map(lambda tk: tk.type_, scan(input_))) == [expected_1, expected_2, TokenType.EOF]

    @pytest.mark.parametrize(
        "input_1, expected_1",
        [
            ("\n", []),
            ("", []),
            (" ", []),
            (" \n", []),
            ("\t", []),
            ("< .", [TokenType.LT, TokenType.DOT]),
            (
                " < .",
                [
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DOT,
                    TokenType.DEDENT,
                ],
            ),
            (
                "  < .",
                [
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DOT,
                    TokenType.DEDENT,
                ],
            ),
            (
                "   < .",
                [
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DOT,
                    TokenType.DEDENT,
                ],
            ),
            (
                "    < .",
                [
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DOT,
                    TokenType.DEDENT,
                ],
            ),
            (
                "\t< .",
                [
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DOT,
                    TokenType.DEDENT,
                ],
            ),
            (
                "<\n\t<\n<",
                [
                    TokenType.LT,
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DEDENT,
                    TokenType.LT,
                ],
            ),
            (
                "<\n <\n<",
                [
                    TokenType.LT,
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DEDENT,
                    TokenType.LT,
                ],
            ),
            (
                "<\n <\n  <\n",
                [
                    TokenType.LT,
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DEDENT,
                    TokenType.DEDENT,
                ],
            ),
            (
                "<\n <\n  <\n<",
                [
                    TokenType.LT,
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.INDENT,
                    TokenType.LT,
                    TokenType.DEDENT,
                    TokenType.DEDENT,
                    TokenType.LT,
                ],
            ),
            ("test", [TokenType.IDENTIFIER]),
            ("\ttest", [TokenType.INDENT, TokenType.IDENTIFIER, TokenType.DEDENT]),
        ],
    )
    def test_manual(self, input_1: str, expected_1: list[TokenType]):
        res = list(map(lambda tk: tk.type_, scan(input_1)))
        assert res == expected_1 + [TokenType.EOF]
        assert res.count(TokenType.INDENT) == res.count(TokenType.DEDENT)
