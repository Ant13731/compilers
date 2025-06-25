from typing import ClassVar

from src.mod.config import debug_print
from src.mod import ast_
from src.mod.optimizer.base_types import RuleVar, Substitution, RewriteRule
from src.mod.optimizer.base_rewrite_functions import match, substitute
from src.mod.optimizer.base_matching_phase import MatchingPhase


def collection_optimizer(ast: ast_.ASTNode, matching_phases: list[MatchingPhase]) -> ast_.ASTNode:
    for matching_phase in matching_phases:
        debug_print(f"Applying matching phase: {matching_phase.__class__.__name__}")
        ast = matching_phase.normalize(ast)
    return ast
