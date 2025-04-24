from lark import Transformer, Token, Tree


class ASTTransformer(Transformer):
    """
    A class to transform a Lark parse tree into an Abstract Syntax Tree (AST).
    """

    def INT(self, token: Token) -> int:
        return int(token.value)

    # def assignment(self, tokens: list[Token]) ->
