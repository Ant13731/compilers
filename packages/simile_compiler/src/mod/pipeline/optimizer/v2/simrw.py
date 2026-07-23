from __future__ import annotations
from typing import Callable, Sequence
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger
from lark import Lark, Transformer, Token
from lark.indenter import PythonIndenter

from src.mod.data import ast_
from src.mod.pipeline.parser import parse


@dataclass
class SimrwRewriteRule:
    name: str
    vars_: list[str]
    rewrite_left: str
    rewrite_right: str
    when: list[str]


class RewriteTransformer(Transformer):
    def start(self, items) -> list[SimrwRewriteRule]:
        return items

    def rule(self, items: tuple[str, tuple[list[str], str, str, list[str]]]) -> SimrwRewriteRule:
        return SimrwRewriteRule(
            items[0],
            items[1][0],
            items[1][1],
            items[1][2],
            items[1][3],
        )

    def vars_and_rest(self, items) -> tuple[list[str], str, str, list[str]]:
        return items[0], *items[1]

    def rewrite_and_rest(self, items) -> tuple[str, str, list[str]]:
        print(f"rewrite_and_rest: {items}")
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


def parse_simrw_file(file_path: str) -> list[SimrwRewriteRule]:
    parser = Lark.open(
        "simrw.lark",
        rel_to=__file__,
        parser="lalr",
        transformer=RewriteTransformer(),
        postlex=PythonIndenter(),
    )
    with open(file_path, "r") as f:
        content = f.read()
    simrw_rewrite_rules: list[SimrwRewriteRule] = parser.parse(content)  # type: ignore
    return simrw_rewrite_rules


@dataclass
class RewriteRule:
    name: str
    vars_: dict[ast_.Identifier, ast_.Type_ | ast_.None_]
    rewrite_left: ast_.ASTNode
    rewrite_right: ast_.ASTNode
    # These should be guarding functions that accept vars_
    when: list[Callable[[list[ast_.ASTNode]], bool]]


# Maps function names in if-statements to guard functions that can block a rewrite rule from being applied
# The key only contains the name of the function - the exact arguments should be parsed from the when_condition string
# and applied to the right typed_vars
GUARD_CONDITIONS: dict[str, Callable[[dict[ast_.Identifier, ast_.Type_ | ast_.None_]], bool]] = {}


def map_to_guard_condition(
    condition_str: str,
    typed_vars: dict[ast_.Identifier, ast_.Type_ | ast_.None_],
) -> Callable[[list[ast_.ASTNode]], bool]:
    return lambda typed_vars: True  # Placeholder for actual implementation


def unwrap_start_nodes(ast_node: ast_.ASTNode) -> ast_.ASTNode:
    if isinstance(ast_node, ast_.Start):
        return unwrap_start_nodes(list(ast_node.children())[0])
    if isinstance(ast_node, ast_.Statements) and len(list(ast_node.children())) == 1:
        return unwrap_start_nodes(list(ast_node.children())[0])
    if isinstance(ast_node, ast_.Assignment):
        return unwrap_start_nodes(ast_node.target)
    match ast_node:
        case ast_.Start(ast_.Statements([ast_.Assignment(target, _, _)]), _):
            return target
        # case ast_.Start(ast_.Statements([child]), _):
        #     return child
        case _:
            return ast_node


def convert_simrw_to_rewrite_rules(simrw_rewrite_rules: list[SimrwRewriteRule]) -> list[RewriteRule]:
    rewrite_rules: list[RewriteRule] = []
    for simrw_rule in simrw_rewrite_rules:
        logger.debug(f"Converting simrw rule {simrw_rule.name} to rewrite rule: {simrw_rule}")
        typed_vars: dict[ast_.Identifier, ast_.Type_ | ast_.None_] = {}
        for simrw_rule_var in simrw_rule.vars_:
            simrw_rule_var += " := 0"  # TODO fix hack: Need to trick the parser into thinking this is a typed assignment, so we can parse it into a TypedName
            typed_var_ast = parse(simrw_rule_var)
            unwrapped_var_ast = unwrap_start_nodes(typed_var_ast)
            if not isinstance(unwrapped_var_ast, ast_.TypedName):
                raise ValueError(f"Expected a TypedName AST node, got {type(unwrapped_var_ast)}")
            if not isinstance(unwrapped_var_ast.name, ast_.Identifier):
                raise ValueError(f"Expected a Identifier AST node on the left side of TypedName, got {type(unwrapped_var_ast)}")
            typed_vars[unwrapped_var_ast.name] = unwrapped_var_ast.type_

        rewrite_left_ast = parse(simrw_rule.rewrite_left)
        rewrite_right_ast = parse(simrw_rule.rewrite_right)

        when_conditions = [map_to_guard_condition(condition_str, typed_vars) for condition_str in simrw_rule.when]
        rewrite_rule = RewriteRule(
            name=simrw_rule.name,
            vars_=typed_vars,
            rewrite_left=rewrite_left_ast,
            rewrite_right=rewrite_right_ast,
            when=when_conditions,
        )
        rewrite_rules.append(rewrite_rule)
    return rewrite_rules
