from __future__ import annotations
from dataclasses import dataclass

from src.mod import ast_


@dataclass
class GeneratorSelection(ast_.ASTNode):

    generator: ast_.In | ast_.BinaryOp
    predicates: ast_.And

    def flatten(self) -> ast_.ListOp:
        """Flatten the generator selection AST into a list of assignments and a condition."""
        return ast_.ListOp.flatten_and_join(
            [
                self.generator,
                self.predicates,
            ],
            ast_.ListOperator.AND,
        )

    def copy_and_concat_predicates(self, new_conditions: ast_.ASTNode | None) -> GeneratorSelection:
        if new_conditions is None:
            return self

        return GeneratorSelection(
            generator=self.generator,
            predicates=ast_.And.flatten_and_join(
                [self.predicates, new_conditions],
                ast_.ListOperator.AND,
            ),
        )

    def _pretty_print_algorithmic(self, indent: int) -> str:
        return self.flatten()._pretty_print_algorithmic(indent)


# @dataclass
# class GeneratorSelectionWithSubstitution(ast_.ASTNode):
#     generator: ast_.In | ast_.BinaryOp
#     substitution: ast_.Equal
#     predicates: ast_.And

#     def flatten(self) -> ast_.ListOp:
#         """Flatten the generator selection AST into a list of assignments and a condition."""
#         return ast_.ListOp.flatten_and_join(
#             [
#                 self.generator,
#                 self.substitution,
#                 self.predicates,
#             ],
#             ast_.ListOperator.AND,
#         )

#     def orient_substitution(self) -> ast_.Equal:
#         """Ensure the LHS of the target of each substitution is an identifier."""

#         if isinstance(self.substitution.left, ast_.Identifier):
#             return self.substitution
#         elif isinstance(self.substitution.right, ast_.Identifier):
#             return ast_.Equal(self.substitution.right, self.substitution.left)
#         else:
#             raise ValueError(f"Substitution {self.substitution} does not have an identifier on either side.")

#     def _pretty_print_algorithmic(self, indent: int) -> str:
#         return self.flatten()._pretty_print_algorithmic(indent)
