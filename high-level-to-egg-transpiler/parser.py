import lark
from lark.indenter import PythonIndenter

from transformer_v1 import GrammarToEggTransformer
from transformer_v2 import EggASTTransformer
from ast_ import BaseEggAST
import ast_
import sys
import inspect


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
    with open("high-level-to-egg-transpiler/very_high_level_grammar.lark", "r") as f:
        grammar = f.read()
        parser = lark.Lark(grammar, start="start", postlex=PythonIndenter(), maybe_placeholders=True)

    # Parse the input string and return the result
    tree = parser.parse(input_string)
    print(tree.pretty())
    # lark.tree.pydot__tree_to_png(tree, f"high-level-to-egg-transpiler/generated_images/tree{i}.png")
    # ast = GrammarToEggTransformer().transform(tree)
    ast: BaseEggAST = EggASTTransformer().transform(tree)
    print(ast)
    print(ast.to_s_expr())


def generate_egg_constructs():
    skip_names = ["UnaryOp", "BinOp"]
    for name, cls_ in inspect.getmembers(ast_):
        if name in skip_names:
            continue
        if inspect.isclass(cls_) and issubclass(cls_, BaseEggAST):
            print(f"Class: {name}, Abstract S-expression: {cls_.to_abstract_s_expr()}")


generate_egg_constructs()


def testing_transformer():
    test_str = """
    1
    1.0
    "hello"
    None
    True
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
    z: bool = a in c
    j: list = [for i in c: i*i]
    k: dict = {for i in c: (i,i*i)}
    l: set = {for i in c: i*i}
    a => b
    m: bool = a == b => e
    a != b and b > a
    b < a and b >= a
    a is a or a is not b
    b <= a or a not in g
    not e != e
    not e is not (not e)
    a + (- (+ b)) - c * d / e % f // g
    a + b - c * d / e % f // g
    a()
    a(1, 2, 3)
    a[]
    a[:]
    a[::]
    a[1:]
    a[:1]
    a[::1]
    a[:1:1]
    a[1::1]
    a[1:2:3]
    j: list = [for i in c | i < 0 and i == 0: i*i]
    k: dict = {for i in c| i < 0: (i,i*i)}
    l: set = {for i in c| i < 0: i*i}
    i^2
    a.b.c
    lambda (a: int,b:int): a + b
    lambda (): a + b
    return a
    return
    break
    continue
    """
    test_strs = test_str.split("\n")
    for i, t in enumerate(test_strs):
        print(f"Parsing string {i}: {t}")
        parse(t, i)

    test_compound_strs = [
        """
if a: b
""",
        """
if a:
    b
""",
        """
if a:
    if b:
        c
    else:
        d
elif a and b:
    if c:
        d
else:
    e
""",
        """
for i in a: b
""",
        """
for i in a:
    for j in b:
        c
""",
        """
struct a:
    a: int
    b: str
""",
        """
enum a:
    a
    b
""",
        """
struct a:
    pass
""",
        """
enum a:
    pass
""",
        """
def a(b: int, c: str) -> int:
    return b + c
""",
    ]
    for i, t in enumerate(test_compound_strs):
        print(f"Parsing string {i}: {t}")
        parse(t, i)
