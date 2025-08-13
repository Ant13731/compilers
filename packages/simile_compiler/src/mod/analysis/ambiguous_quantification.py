from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
import pathlib
from typing import TypeVar

from src.mod.scanner import Location, KEYWORD_TABLE
from src.mod.parser import parse, ParseError
from src.mod import ast_
from src.mod.ast_.symbol_table_types import (
    SimileType,
    ModuleImports,
    ProcedureTypeDef,
    StructTypeDef,
    EnumTypeDef,
    SimileTypeError,
    BaseSimileType,
)

T = TypeVar("T", bound=ast_.ASTNode)


def populate_bound_identifiers(ast: ast_.ASTNode) -> None:
    """Attempts to infer the bound variables of implicitly-bound quantifiers"""
    if isinstance(ast, ast_.Quantifier) and ast._bound_identifiers == set():
        possible_generators = list(filter(lambda x: x.op_type == ast_.BinaryOperator.IN, ast.predicate.find_all_instances(ast_.BinaryOp)))
        possible_bound_identifiers: list[ast_.Identifier | ast_.BinaryOp] = []
        possible_bound_identifier_names: list[ast_.Identifier] = []
        for possible_generator in possible_generators:
            if isinstance(possible_generator.left, ast_.Identifier):
                possible_bound_identifier_names.append(possible_generator.left)
                possible_bound_identifiers.append(possible_generator.left)

            if (
                isinstance(possible_generator.left, ast_.BinaryOp)
                and possible_generator.left.op_type == ast_.BinaryOperator.MAPLET
                and isinstance(possible_generator.left.left, ast_.Identifier)
                and isinstance(possible_generator.left.right, ast_.Identifier)
            ):
                possible_bound_identifier_names.append(possible_generator.left.left)
                possible_bound_identifier_names.append(possible_generator.left.right)
                possible_bound_identifiers.append(ast_.Maplet(possible_generator.left.left, possible_generator.left.right))

        for possible_bound_identifier in possible_bound_identifier_names:
            assert ast._env is not None
            if ast._env.get(possible_bound_identifier.name) is not None:
                possible_bound_identifiers = list(filter(lambda x: not x.contains_item(possible_bound_identifier), possible_bound_identifiers))

        if not possible_bound_identifiers:
            raise SimileTypeError(
                f"Failed to infer bound variables for quantifier {ast.pretty_print_algorithmic()}. "
                "Either the expression is ambiguously overwriting a predefined variable in scope, "
                "or no valid generators are present in the quantification expression. Please explicitly state bound variables",
                ast,
            )

        ast._bound_identifiers = set(possible_bound_identifiers)

    for child in ast.children(True):
        populate_bound_identifiers(child)
