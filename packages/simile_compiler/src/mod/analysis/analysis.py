from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
from typing import TypeVar

from src.mod import ast_
from src.mod.analysis.populate_ast_environments import populate_ast_environments
from src.mod.analysis.reserved_keywords import reserved_keywords_check
from src.mod.analysis.ambiguous_quantification import populate_bound_identifiers
from src.mod.analysis.type_analysis import type_check


T = TypeVar("T", bound=ast_.ASTNode)


def semantic_analysis(ast: T) -> T:
    ast = populate_ast_environments(ast)
    ast = reserved_keywords_check(ast)
    populate_bound_identifiers(ast)
    type_check(ast)
    return ast
