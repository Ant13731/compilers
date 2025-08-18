from __future__ import annotations
from dataclasses import dataclass

from src.mod import ast_


@dataclass
class GeneratorSelection(ast_.ASTNode):

    generator: ast_.In | ast_.BinaryOp
    predicates: ast_.And

    def flatten(self) -> ast_.ListOp:
        """Flatten the generator selection AST into a list of assignments and a condition."""
        return ast_.And(
            [
                self.generator,
                self.predicates,
            ],
        )

    def copy_and_concat_predicates(self, new_conditions: ast_.ASTNode | None) -> GeneratorSelection:
        if new_conditions is None:
            return self

        return GeneratorSelection(
            generator=self.generator,
            predicates=ast_.And(
                [
                    self.predicates,
                    new_conditions,
                ],
            ),
        )

    def _pretty_print_algorithmic(self, indent: int) -> str:
        return self.flatten()._pretty_print_algorithmic(indent)


@dataclass
class GeneratorSelectionV2(ast_.ASTNode):
    bound_identifiers: set[ast_.Identifier | ast_.MapletIdentifier]
    generators: list[ast_.In]
    predicates: ast_.And


@dataclass
class CombinedGeneratorSelectionV2(ast_.ASTNode):
    bound_identifier: ast_.Identifier | ast_.MapletIdentifier
    generator: ast_.In
    gsp_predicates: ast_.Or  # Or[GeneratorSelectionV2]
    predicates: ast_.And | None = None


@dataclass
class SingleGeneratorSelectionV2(ast_.ASTNode):
    bound_identifier: ast_.Identifier | ast_.MapletIdentifier
    generator: ast_.In
    predicates: ast_.And


@dataclass
class Loop(ast_.ASTNode):
    predicate: ast_.Or | GeneratorSelectionV2 | CombinedGeneratorSelectionV2 | SingleGeneratorSelectionV2
    body: ast_.ASTNode


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
