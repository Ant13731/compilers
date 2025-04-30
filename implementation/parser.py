import pprint

import lark
from lark.indenter import PythonIndenter

from transformer import EggASTTransformer
from ast_ import BaseAST, Start


def parse(input_string: str, start_rule: str = "start") -> BaseAST | Start:
    """
    Parse the input string using Lark parser.

    Args:
        input_string (str): The input string to parse.
        start_rule (str): The starting rule for the parser. Default is "start".

    Returns:
        BaseAST: Parsed output in AST form.
    """

    # Add a newline at the end of the input string to prevent EOF errs
    input_string = input_string + "\n"

    # Create a Lark parser with the defined grammar
    with open("implementation/grammar.lark", "r") as f:
        grammar = f.read()
        parser = lark.Lark(grammar, start="start", postlex=PythonIndenter(), maybe_placeholders=True)

    # Parse the input string and return the result
    tree = parser.parse(input_string, start=start_rule)
    ast: BaseAST = EggASTTransformer().transform(tree)
    return ast
