from dataclasses import dataclass
from typing import Callable, Self

from src.mod import ast_


@dataclass
class RuleVar(ast_.ASTNode):
    """For defining variables in matching/rewrite rules"""

    id: str

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuleVar):
            return False
        return self.id == other.id


Substitution = dict[RuleVar, ast_.ASTNode]
PreMatchedASTNode = ast_.ASTNode


@dataclass
class RewriteRule:
    lh: ast_.ASTNode
    rh: ast_.ASTNode
    match_condition: Callable[[Self, PreMatchedASTNode], bool] | None = None
    substitution_condition: Callable[[Self, Substitution, PreMatchedASTNode], bool] | None = None
