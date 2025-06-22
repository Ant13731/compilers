from dataclasses import dataclass
from typing import ClassVar, Callable, Self

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


class MatchingPhase:
    """Each matching phase contains a list of rules that are applied to the AST in order exhaustively, starting from the root.

    Exit conditions separate these phases, allowing for early termination of the matching process."""

    rules: ClassVar[list[RewriteRule]]

    @classmethod
    def exit_condition(cls, ast: ast_.ASTNode) -> bool:
        raise NotImplementedError
