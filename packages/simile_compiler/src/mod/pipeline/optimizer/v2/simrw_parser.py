from __future__ import annotations
from typing import Callable, Sequence
from dataclasses import dataclass, field
from copy import deepcopy
import pathlib

from loguru import logger
from lark import Lark, Transformer, Token
from lark.indenter import PythonIndenter

from src.mod.data import ast_
from src.mod.pipeline.parser import parse


@dataclass
class SimrwAST:
    name: str
    vars_: list[str]
    rewrite_left: str
    rewrite_right: str
    when: list[str]


class RewriteTransformer(Transformer):
    def start(self, items) -> list[SimrwAST]:
        return items

    def rule(self, items: tuple[Token, tuple[list[str], str, str, list[str]]]) -> SimrwAST:
        return SimrwAST(
            items[0].value,
            items[1][0],
            items[1][1],
            items[1][2],
            items[1][3],
        )

    def vars_and_rest(self, items) -> tuple[list[str], str, str, list[str]]:
        return items[0], *items[1]

    def rewrite_and_rest(self, items) -> tuple[str, str, list[str]]:
        if items[2] is None:
            return items[0], items[1], []
        return items[0], items[1], items[2]

    def when_and_rest(self, items: list[list[str]]) -> list[str]:
        return items[0]

    def vars(self, items) -> list[str]:
        return [str(i).strip() for i in items]

    def rewrite_left(self, items) -> str:
        return "\n".join([str(i).strip() for i in items])

    def rewrite_right(self, items: list[Token]) -> str:
        return "\n".join([str(i).strip() for i in items])

    def when(self, items: list[Token]) -> list[str]:
        return [str(i).strip() for i in items]


def parse_simrw_file(file_path: str | pathlib.Path) -> list[SimrwAST]:
    parser = Lark.open(
        "simrw.lark",
        rel_to=__file__,
        parser="lalr",
        transformer=RewriteTransformer(),
        postlex=PythonIndenter(),
    )
    with open(file_path, "r") as f:
        content = f.read()
    simrw_rewrite_rules: list[SimrwAST] = parser.parse(content)  # type: ignore
    return simrw_rewrite_rules
