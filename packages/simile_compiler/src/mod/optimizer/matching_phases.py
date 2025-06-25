from dataclasses import dataclass
from typing import ClassVar

from src.mod import ast_
from src.mod.optimizer.base_types import RewriteRule, RuleVar
from src.mod.optimizer.base_matching_phase import MatchingPhase


@dataclass
class SetsToComprehensionPhase(MatchingPhase):
    rules: ClassVar[list[RewriteRule]] = [
        RewriteRule(
            lh=RuleVar("S"),
            rh=ast_.SetComprehension(
                ast_.IdentList([ast_.Identifier("x")]),
                ast_.In(ast_.Identifier("x"), RuleVar("S")),
                ast_.Identifier("x"),
            ),
            match_condition=lambda self, ast: isinstance(ast.get_type, ast_.SetType),
        ),
        RewriteRule(
            lh=RuleVar("S"),
            rh=ast_.SetComprehension(
                ast_.IdentList([ast_.Identifier("x")]),
                ast_.In(ast_.Identifier("x"), RuleVar("S")),
                ast_.Identifier("x"),
            ),
            match_condition=lambda self, ast: isinstance(ast, ast_.Enumeration),
        ),
    ]

    def exit_condition(self, ast: ast_.ASTNode) -> bool:
        return self.num_of_ast_repeats != 0
