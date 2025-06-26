# from dataclasses import dataclass
# from typing import ClassVar

# from src.mod import ast_
# from src.mod import analysis
# from src.mod.optimizer.base_types import RewriteRule, RuleVar
# from src.mod.optimizer.base_matching_phase import MatchingPhase


# @dataclass
# class SetsToComprehensionPhase(MatchingPhase):
#     rules: ClassVar[list[RewriteRule]] = [
#         # RewriteRule(
#         #     lh=RuleVar("S"),
#         #     rh=ast_.SetComprehension(
#         #         ast_.IdentList([ast_.Identifier("x")]),
#         #         ast_.In(ast_.Identifier("x"), RuleVar("S")),
#         #         ast_.Identifier("x"),
#         #     ),
#         #     match_condition=lambda ast: isinstance(ast.get_type, ast_.SetType),
#         # ),
#         # RewriteRule(
#         #     lh=RuleVar("S"),
#         #     rh=ast_.SetComprehension(
#         #         ast_.IdentList([ast_.Identifier("x")]),
#         #         ast_.In(ast_.Identifier("x"), RuleVar("S")),
#         #         ast_.Identifier("x"),
#         #     ),
#         #     match_condition=lambda ast: isinstance(ast, ast_.Enumeration),
#         # ),
#         RewriteRule(
#             lh=ast_.BinaryOp(ast_.Identifier(RuleVar("S")), RuleVar("T"), RuleVar("op")),
#             rh=ast_.SetComprehension(
#                 ast_.IdentList([ast_.Identifier("x")]),
#                 ast_.In(ast_.Identifier("x"), ast_.Identifier(RuleVar("S"))),
#                 ast_.BinaryOp(ast_.Identifier("x"), RuleVar("op"), ast_.Identifier(RuleVar("T"))),
#             ),
#         )
#     ]

#     def exit_condition(self, ast: ast_.ASTNode) -> bool:
#         return self.num_of_ast_repeats != 0


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
#         # ast_.Identifier("s"),
#     ]
# )
# print(TEST.pretty_print())
# TEST_TYPE = analysis.populate_ast_environments(TEST)
# TEST_PHASE = SetsToComprehensionPhase()
# print(TEST_PHASE.normalize(TEST_TYPE).pretty_print())
