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


# TODO remove
# comprehension_test = r"""
# {x | x in A and x in B and -1 < x < 0}
# """
# print("Comprehension test:")
# parse(comprehension_test)


# # generate_egg_constructs()


# # TODO: update old tests for comprehensions
# def testing_transformer():
#     test_str = r"""
#     1
#     1.0
#     "hello"
#     None
#     True
#     a: int = 1
#     a: int = (1)
#     b: str = "hello"
#     c: list = [1, 2, 3]
#     d: dict = {"key": "value"}
#     e: bool = True
#     f: float = 3.14
#     g: set = {1, 2, 3}
#     h: tuple = (1, 2, 3)
#     i: None = None
#     z: bool = a in c
#     j: list = [for i in c: i*i]
#     k: dict = {for i in c: (i,i*i)}
#     l: set = {for i in c: i*i}
#     a ==> b
#     m: bool = a == b ==> e
#     a != b and b > a
#     b < a and b >= a
#     a is a or a is not b
#     b <= a or a not in g
#     not e != e
#     not e is not (not e)
#     a + (- (+ b)) - c * d / e % f // g
#     a + b - c * d / e % f // g
#     a()
#     a(1, 2, 3)
#     a[]
#     a[:]
#     a[::]
#     a[1:]
#     a[:1]
#     a[::1]
#     a[:1:1]
#     a[1::1]
#     a[1:2:3]
#     j: list = [for i in c | i < 0 and i == 0: i*i]
#     k: dict = {for i in c| i < 0: (i,i*i)}
#     l: set = {for i in c| i < 0: i*i}
#     i^2
#     a.b.c
#     lambda (a: int,b:int): a + b
#     lambda (): a + b
#     return a
#     return
#     break
#     continue
#     a \subset b
#     a \subseteq b
#     a \supset b
#     a \supseteq b
#     a \cup b
#     a \cap b
#     a \setminus b
#     a \ctimes b
#     a \circ b
#     \powerset a
#     ~a
#     ~a \cup ~b
#     a ==> b ==> c
#     c <== a <== b
#     a <==> b
#     a <!==> b
#     """
#     test_strs = test_str.split("\n")
#     for i, t in enumerate(test_strs):
#         print(f"Parsing string {i}: {t}")
#         parse(t, i)

#     test_compound_strs = [
#         """
# if a: b
# """,
#         """
# if a:
#     b
# """,
#         """
# if a:
#     if b:
#         c
#     else:
#         d
# elif a and b:
#     if c:
#         d
# else:
#     e
# """,
#         """
# for i in a: b
# """,
#         """
# for i in a:
#     for j in b:
#         c
# """,
#         """
# struct a:
#     a: int
#     b: str
# """,
#         """
# enum a:
#     a
#     b
# """,
#         """
# struct a:
#     pass
# """,
#         """
# enum a:
#     pass
# """,
#         """
# def a(b: int, c: str) -> int:
#     return b + c
# """,
#     ]
#     for i, t in enumerate(test_compound_strs):
#         print(f"Parsing string {i}: {t}")
#         parse(t, i)


# # testing_transformer()
