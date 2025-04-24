import lark
from lark.indenter import PythonIndenter

from ast_ import ASTTransformer


def parse(input_string: str, i: int):
    """
    Parse the input string using Lark parser.

    Args:
        input_string (str): The input string to parse.

    Returns:
        str: The parsed output.
    """

    # Add a newline at the end of the input string to prevent EOF errs
    input_string = input_string + "\n"

    # Create a Lark parser with the defined grammar
    with open("src/very_high_level_grammar.lark", "r") as f:
        grammar = f.read()
        parser = lark.Lark(grammar, start="start", postlex=PythonIndenter())

    # Parse the input string and return the result
    tree = parser.parse(input_string)
    print(tree.pretty())
    lark.tree.pydot__tree_to_png(tree, f"src/generated_images/tree{i}.png")
    ast = ASTTransformer().transform(tree)
    print(ast)


test_str = """
a: int = 1
a: int = (1)
b: str = "hello"
c: list = [1, 2, 3]
d: dict = {"key": "value"}
e: bool = True
f: float = 3.14
g: set = {1, 2, 3}
h: tuple = (1, 2, 3)
i: None = None
j: list = [for i in c: i*i]
k: dict = {for i in c: (i,i*i)}
l: set = {for i in c: i*i}
"""
test_strs = test_str.split("\n")
[parse(t, i) for i, t in enumerate(test_strs[1:2])]
