from __future__ import annotations
from dataclasses import dataclass

from src.mod import ast_


@dataclass
class GeneratorSelectionAST(ast_.ASTNode):
    generator: ast_.In | ast_.BinaryOp
    assignments: list[ast_.Assignment]
    condition: ast_.And

    def flatten(self) -> ast_.ListOp:
        """Flatten the generator selection AST into a list of assignments and a condition."""
        return ast_.ListOp.flatten_and_join(
            [
                self.generator,
                self.condition,
            ]
            + list(map(lambda x: ast_.Equal(x.target, x.value), self.assignments)),
            ast_.ListOperator.AND,
        )

    def with_new_conditions(self, new_conditions: ast_.ASTNode | None) -> GeneratorSelectionAST:
        """Return a new GeneratorSelectionAST with the given new conditions."""
        if new_conditions is None:
            return self

        return GeneratorSelectionAST(
            generator=self.generator,
            assignments=self.assignments,
            condition=ast_.And.flatten_and_join([self.condition, new_conditions], ast_.ListOperator.AND),
        )

    def _pretty_print_algorithmic(self, indent: int) -> str:
        assignments_str = " ∧ ".join(
            [f"{assignment.target._pretty_print_algorithmic(indent)} = {assignment.value._pretty_print_algorithmic(indent)}" for assignment in self.assignments]
        )

        ret = self.generator._pretty_print_algorithmic(indent)
        if assignments_str:
            ret += f" ∧ {assignments_str}"
        if len(self.condition.items) > 0:
            ret += f" ∧ {self.condition._pretty_print_algorithmic(indent)}"
        return ret
