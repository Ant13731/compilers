from typing import ClassVar

from src.mod.optimizer.base_types import MatchingPhase, RewriteRule, RuleVar
from src.mod import ast_


class TestMatchingPhase(MatchingPhase):
    rules: ClassVar[list[RewriteRule]] = [
        RewriteRule(
            ast_.BinaryOp(RuleVar("x"), RuleVar("x"), op_type=ast_.BinaryOperator.MULTIPLY),
            ast_.BinaryOp(RuleVar("x"), ast_.Int("2"), op_type=ast_.BinaryOperator.EXPONENT),
        ),
        RewriteRule(
            ast_.BinaryOp(RuleVar("x"), RuleVar("x"), op_type=ast_.BinaryOperator.ADD),
            ast_.BinaryOp(RuleVar("x"), ast_.Int("2"), op_type=ast_.BinaryOperator.MULTIPLY),
        ),
        # RewriteRule(
        #     ast_.BinaryOp(RuleVar("x"), ast_.Int("2"), op_type=ast_.BinaryOpType.MULTIPLY),
        #     RuleVar("x_modified"),
        #     substitution_condition=helper_substitution_condition,
        # ),
    ]

    @classmethod
    def exit_condition(cls, ast: ast_.ASTNode) -> bool:
        return False  # No exit condition for testing purposes


class SetsToComprehensionPhase(MatchingPhase):
    rules: ClassVar[list[RewriteRule]] = [
        RewriteRule(),
    ]

    @classmethod
    def exit_condition(cls, ast: ast_.ASTNode) -> bool:
        return False  # No exit condition for testing purposes
