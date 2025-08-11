from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection
from src.mod.optimizer.intermediate_ast import GeneratorSelection

# NOTE: REWRITE RULES MUST ALWAYS USE THE PARENT FORM FOR STRUCTURAL MATCHING (ex. BinaryOp instead of Add)


@dataclass
class SyntacticSugarForBags(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.bag_image,
        ]

    def bag_image(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Image(r, b):
                if not isinstance(r.get_type, ast_.SetType) or not ast_.SetType.is_relation(r.get_type):
                    logger.warning(f"First argument to bag image {r} is not a relation")
                    return None
                if not isinstance(b.get_type, ast_.SetType) or not ast_.SetType.is_bag(b.get_type):
                    logger.warning(f"Second argument to bag image {b} is not a bag")
                    return None
                return ast_.Composition(ast_.Inverse(r), b)
        return None


@dataclass
class BuiltinFunctions(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.cardinality,
            self.domain,
            self.range_,
        ]

    def cardinality(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("card"), [s]):
                if not isinstance(s.get_type, ast_.SetType):
                    logger.warning(f"Argument to cardinality {s} is not a set")
                    return None

                fresh_variable = ast_.Identifier(self._get_fresh_identifier_name())
                ast_sum = ast_.Sum(
                    ast_.And([ast_.In(fresh_variable, s)]),
                    ast_.Int("1"),
                )
                ast_sum._bound_identifiers = {fresh_variable}
                return ast_sum
        return None

    def domain(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("dom"), [s]):
                if not isinstance(s.get_type, ast_.SetType):
                    logger.warning(f"Argument to domain {s} is not a set")
                    return None

                left = ast_.Identifier(self._get_fresh_identifier_name())
                right = ast_.Identifier(self._get_fresh_identifier_name())
                fresh_variable = ast_.Maplet(left, right)

                ast_sum = ast_.SetComprehension(
                    ast_.And([ast_.In(fresh_variable, s)]),
                    fresh_variable.left,
                )
                ast_sum._bound_identifiers = {left, right}
                return ast_sum
        return None

    def range_(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("ran"), [s]):
                if not isinstance(s.get_type, ast_.SetType):
                    logger.warning(f"Argument to range {s} is not a set")
                    return None

                left = ast_.Identifier(self._get_fresh_identifier_name())
                right = ast_.Identifier(self._get_fresh_identifier_name())
                fresh_variable = ast_.Maplet(left, right)

                ast_sum = ast_.SetComprehension(
                    ast_.And([ast_.In(fresh_variable, s)]),
                    fresh_variable.right,
                )
                ast_sum._bound_identifiers = {left, right}
                return ast_sum
        return None
