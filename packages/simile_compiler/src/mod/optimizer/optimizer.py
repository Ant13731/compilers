from typing import ClassVar

from src.mod.config import debug_print
from src.mod import ast_

from src.mod.optimizer.rewrite_collection import RewriteCollection


def collection_optimizer(ast: ast_.ASTNode, matching_phases: list[RewriteCollection]) -> ast_.ASTNode:
    for matching_phase in matching_phases:
        debug_print(f"Applying matching phase: {matching_phase.__class__.__name__}")
        ast = matching_phase.normalize(ast)
    return ast
