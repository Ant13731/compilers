# from typing import ClassVar
# from dataclasses import dataclass, fields

# from src.mod.config import debug_print
# from src.mod import ast_
# from src.mod.optimizer.base_types import RuleVar, Substitution, RewriteRule
# from src.mod.optimizer.base_rewrite_functions import match, substitute


# @dataclass
# class MatchingPhase:
#     """Each matching phase contains a list of rules that are applied to the AST in order exhaustively, starting from the root.

#     Exit conditions separate these phases, allowing for early termination of the matching process."""

#     num_of_matches: int = 0
#     num_of_ast_repeats: int = 0
#     rules: ClassVar[list[RewriteRule]]

#     def exit_condition(self, ast: ast_.ASTNode) -> bool:
#         """Exit condition for the matching phase. If this condition is met, the phase will stop applying rules."""
#         return False

#     def apply_all_rules_once(self, ast: ast_.ASTNode) -> ast_.ASTNode:
#         for rewrite_rule in self.rules:
#             if self.exit_condition(ast):
#                 debug_print(f"Exit condition met for phase {self.__class__.__name__}, stopping rule application.")
#                 return ast

#             # First check match condition to determine if we want to attempt rule application
#             if rewrite_rule.match_condition is not None and not rewrite_rule.match_condition(ast):
#                 continue

#             debug_print(f"ATTEMPT: to match rule: {rewrite_rule.lh} -> {rewrite_rule.rh} with AST: {ast}")

#             # Attempt to match the left-hand side (lh) with the AST (ast)
#             substitutions = match(rewrite_rule.lh, ast)

#             if substitutions is None:
#                 debug_print(f"FAILED: to match lh side of rule: {rewrite_rule.lh} with AST: {ast}")
#                 continue

#             if rewrite_rule.substitution_condition is not None and not rewrite_rule.substitution_condition(substitutions, ast):
#                 debug_print(f"SKIPPED: substitution condition for rule {rewrite_rule.rh} with substitutions {substitutions}")
#                 continue

#             # If a match is found, rewrite the AST with the right-hand side (rh)
#             ast = substitute(rewrite_rule.rh, substitutions)
#             self.num_of_matches += 1
#             debug_print(f"SUCCESS: applied substitutions {substitutions} to rule {rewrite_rule.rh}, resulting in new ast: {ast}")
#         return ast

#     def apply_all_rules_once_recursive(self, ast: ast_.ASTNode) -> ast_.ASTNode:
#         new_ast_children: list = []
#         for c in fields(ast):
#             child = getattr(ast, c.name)
#             if isinstance(child, list):
#                 # If the child is a list, we need to apply the rules to each element in the list
#                 new_child_list = []
#                 for item in child:
#                     if not isinstance(item, ast_.ASTNode):
#                         new_child_list.append(item)
#                         continue
#                     item = self.apply_all_rules_once_recursive(item)
#                     new_child_list.append(item)
#                 new_ast_children.append(new_child_list)
#                 continue

#             if not isinstance(child, ast_.ASTNode):
#                 new_ast_children.append(child)
#                 continue

#             # Recursively normalize children first
#             child = self.apply_all_rules_once_recursive(child)
#             new_ast_children.append(child)

#         new_ast = ast.__class__(*new_ast_children)
#         new_ast._env = ast._env
#         debug_print(f"\nNormalizing AST: {ast}")
#         return self.apply_all_rules_once(new_ast)

#     def normalize(self, ast: ast_.ASTNode) -> ast_.ASTNode:
#         """Run the matching phase on the given AST."""
#         prev_num_of_matches = 0

#         while True:
#             prev_num_of_matches = self.num_of_matches
#             ast = self.apply_all_rules_once_recursive(ast)
#             self.num_of_ast_repeats += 1

#             if prev_num_of_matches == self.num_of_matches:
#                 break

#         return ast
