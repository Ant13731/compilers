from __future__ import annotations
from typing import Callable
from functools import wraps
from dataclasses import dataclass, fields

from src.mod.config import debug_print
from src.mod import ast_


@dataclass
class RewriteCollection:
    fresh_variable_counter: int = 0

    num_of_attempted_matches: int = 0
    num_of_matches: int = 0

    num_of_full_ast_traversals: int = 0

    def rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return list(map(self._counter_wrapper, self._rewrite_collection()))

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        raise NotImplementedError

    def _get_fresh_identifier_name(self) -> str:
        self.fresh_variable_counter += 1
        return f"*fresh_var_{self.fresh_variable_counter:05}"

    def _counter_wrapper(self, rewrite_rule: Callable[[ast_.ASTNode], ast_.ASTNode | None]) -> Callable[[ast_.ASTNode], ast_.ASTNode | None]:
        @wraps(rewrite_rule)
        def wrapped_rewrite_rule(ast: ast_.ASTNode) -> ast_.ASTNode | None:
            self.num_of_attempted_matches += 1
            result = rewrite_rule(ast)
            if result is not None:
                self.num_of_matches += 1
            return result

        return wrapped_rewrite_rule

    def apply_all_rules_once(self, ast: ast_.ASTNode) -> ast_.ASTNode:
        """Apply all rewrite rules in the collection to the AST once."""

        for rewrite_rule in self.rewrite_collection():
            debug_print(f"ATTEMPT: to match rule: {rewrite_rule.__name__} with AST: {ast}")

            new_ast = rewrite_rule(ast)
            if new_ast is not None:
                ast = new_ast
                debug_print(f"SUCCESS: matched {rewrite_rule.__name__}, resulting in new ast: {ast}")
                continue

            debug_print(f"FAILED: to match lh side of rule: {rewrite_rule.__name__} with AST: {ast}")

        return ast

    def apply_all_rules_one_traversal(self, ast: ast_.ASTNode) -> ast_.ASTNode:
        """Apply all rewrite rules in the collection to the AST, traversing the AST recursively."""
        new_ast_children: list = []
        for c in fields(ast):
            child = getattr(ast, c.name)
            if isinstance(child, list):
                # If the child is a list, we need to apply the rules to each element in the list
                new_child_list = []
                for item in child:
                    if not isinstance(item, ast_.ASTNode):
                        new_child_list.append(item)
                        continue
                    item = self.apply_all_rules_one_traversal(item)
                    new_child_list.append(item)
                new_ast_children.append(new_child_list)
                continue

            if not isinstance(child, ast_.ASTNode):
                new_ast_children.append(child)
                continue

            # Recursively normalize children first
            child = self.apply_all_rules_one_traversal(child)
            new_ast_children.append(child)

        new_ast = ast.__class__(*new_ast_children)
        new_ast._env = ast._env
        debug_print(f"\nNormalizing AST: {ast}")
        return self.apply_all_rules_once(new_ast)

    def normalize(self, ast: ast_.ASTNode) -> ast_.ASTNode:
        """Run the matching phase on the given AST."""
        prev_num_of_matches = 0

        while True:
            prev_num_of_matches = self.num_of_matches
            ast = self.apply_all_rules_one_traversal(ast)
            self.num_of_full_ast_traversals += 1

            if prev_num_of_matches == self.num_of_matches:
                break

        return ast
