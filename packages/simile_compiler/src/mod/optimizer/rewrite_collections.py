from __future__ import annotations
from typing import Callable
from dataclasses import dataclass

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection


@dataclass
class PhaseOneRewriteCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.rewrite_rule_1,
        ]

    def rewrite_rule_1(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(left, right, ast_.BinaryOperator.UNION):
                if any(
                    [
                        not isinstance(left, ast_.Identifier),
                        not isinstance(left.get_type, ast_.SetType),
                        not isinstance(right.get_type, ast_.SetType),
                    ]
                ):
                    return None
                fresh_name = self._get_fresh_identifier_name()
                new_ast = ast_.SetComprehension(
                    ast_.IdentList([ast_.Identifier(fresh_name)]),
                    ast_.In(ast_.Identifier(fresh_name), left),
                    ast_.Identifier(fresh_name),
                )
                return analysis.add_environments_to_ast(new_ast, ast._env)

        return None


# TEST = ast_.Statements(
#     [
#         ast_.Assignment(
#             ast_.Identifier("s"),
#             ast_.SetEnumeration(
#                 [
#                     ast_.Int("1"),
#                     ast_.Int("2"),
#                     ast_.Int("3"),
#                 ],
#             ),
#         ),
#         ast_.Assignment(
#             ast_.Identifier("s"),
#             ast_.Union(
#                 ast_.Identifier("s"),
#                 ast_.SetEnumeration(
#                     [
#                         ast_.Int("4"),
#                         ast_.Int("5"),
#                     ],
#                 ),
#             ),
#         ),
#     ]
# )
# print(TEST.pretty_print())
# TEST_TYPE = analysis.populate_ast_environments(TEST)
# print(TEST_TYPE.pretty_print(print_env=True))
# TEST_PHASE = PhaseOneRewriteCollection()
# print(TEST_PHASE.normalize(TEST_TYPE).pretty_print(print_env=True))
