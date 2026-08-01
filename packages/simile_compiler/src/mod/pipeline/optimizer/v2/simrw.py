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


def simrw_to_source(rewrite_rule: RewriteRule) -> str:
    ret = f"rule {rewrite_rule.name}:\n"
    ret += "\tvars:\n"
    if rewrite_rule.vars_:
        ret += "\t\t"
        ret += "\n\t\t".join(f"{ast_.ast_to_source(var)}: {ast_.ast_to_source(type_)}" for var, type_ in rewrite_rule.vars_.items())
        ret += "\n"
    ret += f"\trewrite:\n"
    ret += f"\t\t{ast_.ast_to_source(rewrite_rule.rewrite_left)}\n"
    ret += "\t\t~>\n"
    ret += f"\t\t{ast_.ast_to_source(rewrite_rule.rewrite_right)}\n"
    if rewrite_rule.when:
        ret += "\twhen:\n"
        ret += "\t\t"
        ret += "\n\t\t".join(f"{condition.__name__}" for condition in rewrite_rule.when)
        ret += "\n"
    return ret


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
    match ast_node:
        case ast_.Start(body, _):
            return unwrap_start_nodes(body)
        case ast_.Statements([child]):
            return unwrap_start_nodes(child)
    return ast_node


def convert_simrw_to_rewrite_rules(rewrite_rule_asts: list[SimrwAST]) -> list[RewriteRule]:
    rewrite_rules: list[RewriteRule] = []
    for rewrite_rule_ast in rewrite_rule_asts:
        logger.debug(
            f"Converting simrw rule {rewrite_rule_ast.name} to rewrite rule (with vars {rewrite_rule_ast.vars_}): {rewrite_rule_ast.rewrite_left} ~> {rewrite_rule_ast.rewrite_right}"
        )

        typed_vars: dict[ast_.Identifier, ast_.Type_ | ast_.None_] = {}
        for simrw_rule_var in rewrite_rule_ast.vars_:
            typed_var_ast = parse(simrw_rule_var)
            unwrapped_var_ast = unwrap_start_nodes(typed_var_ast)
            if not isinstance(unwrapped_var_ast, ast_.TypedName):
                raise ValueError(f"Expected a TypedName AST node, got {type(unwrapped_var_ast)}")
            if not isinstance(unwrapped_var_ast.name, ast_.Identifier):
                raise ValueError(f"Expected a Identifier AST node on the left side of TypedName, got {type(unwrapped_var_ast)}")
            typed_vars[unwrapped_var_ast.name] = unwrapped_var_ast.type_

        rewrite_left_ast = parse(rewrite_rule_ast.rewrite_left)
        rewrite_right_ast = parse(rewrite_rule_ast.rewrite_right)

        when_conditions = [map_to_guard_condition(condition_str, typed_vars) for condition_str in rewrite_rule_ast.when]
        rewrite_rule = RewriteRule(
            name=rewrite_rule_ast.name,
            vars_=typed_vars,
            rewrite_left=rewrite_left_ast,
            rewrite_right=rewrite_right_ast,
            when=when_conditions,
        )
        rewrite_rules.append(rewrite_rule)
    return rewrite_rules
