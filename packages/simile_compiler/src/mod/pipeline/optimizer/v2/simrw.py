from __future__ import annotations
from typing import Callable, Sequence
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from src.mod.data import ast_
from src.mod.pipeline.parser import parse
from src.mod.pipeline.optimizer.v2.simrw_parser import SimrwAST


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


def convert_simrw_to_rewrite_rules(simrw_rewrite_rules: list[SimrwAST]) -> list[RewriteRule]:
    rewrite_rules: list[RewriteRule] = []
    for simrw_rule in simrw_rewrite_rules:
        logger.debug(f"Converting simrw rule {simrw_rule.name} to rewrite rule: {simrw_rule}")
        typed_vars: dict[ast_.Identifier, ast_.Type_ | ast_.None_] = {}
        for simrw_rule_var in simrw_rule.vars_:
            # simrw_rule_var += " := 0"  # TODO fix hack: Need to trick the parser into thinking this is a typed assignment, so we can parse it into a TypedName
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
