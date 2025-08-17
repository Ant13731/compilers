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


@dataclass
class SetComprehensionConstructionCollection(RewriteCollection):
    bound_quantifier_variables: set[ast_.Identifier] = field(default_factory=set)
    # current_bound_identifiers: list[set[ast_.Identifier | ast_.MapletIdentifier]] = field(default_factory=list)

    def apply_all_rules_one_traversal(self, ast):
        # Before entering a new quantifier, record currently bound variables
        # (so we can restore them later)
        bound_quantifier_variables_before = deepcopy(self.bound_quantifier_variables)
        if isinstance(ast, ast_.Quantifier):
            logger.debug(f"Quantifier found with bound variables: {ast.bound}")
            # Add newly accessible (bound) variables, used for generator checks in membership collapse
            self.bound_quantifier_variables |= ast.bound
            # self.current_bound_identifiers.append(ast._bound_identifiers)

        ast = super().apply_all_rules_one_traversal(ast)
        logger.debug(f"AST after applying all rules: {ast.pretty_print_algorithmic()}")

        # Restore bound variable record since we have exited the possibly nested quantifier
        self.bound_quantifier_variables = bound_quantifier_variables_before

        # if self.current_bound_identifiers and hasattr(ast, "_bound_identifiers") and ast._bound_identifiers != self.current_bound_identifiers:
        #     # If the current bound identifiers are set, we need to update the AST's bound identifiers
        #     ast._bound_identifiers = self.current_bound_identifiers.pop()
        logger.debug(f"AST after swapping bound identifiers: {ast.pretty_print_algorithmic()}")

        return ast

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.predicate_operations,
            # self.singleton_membership,
            self.membership_collapse,
        ]

    def predicate_operations(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # Idea, if we want to add some compilation-time optimizations (like union of two set enums into one set enum),
            # we can just add those rules here
            case ast_.BinaryOp(left, right, op_type) if op_type in (
                ast_.BinaryOperator.UNION,
                ast_.BinaryOperator.INTERSECTION,
                ast_.BinaryOperator.DIFFERENCE,
            ):
                if any(
                    [
                        not isinstance(left.get_type, ast_.SetType),
                        not isinstance(right.get_type, ast_.SetType),
                    ]
                ):
                    logger.debug(f"FAILED: at least one union child is not a set type: {left.get_type}, {right.get_type}")
                    return None
                fresh_name = self._get_fresh_identifier_name()

                match op_type:
                    case ast_.BinaryOperator.UNION:
                        list_op: type[ast_.And | ast_.Or] = ast_.Or
                        right_join_op: type[ast_.In | ast_.NotIn] = ast_.In
                    case ast_.BinaryOperator.INTERSECTION:
                        list_op = ast_.And
                        right_join_op = ast_.In
                    case ast_.BinaryOperator.DIFFERENCE:
                        list_op = ast_.And
                        right_join_op = ast_.NotIn

                new_ast = ast_.SetComprehension(
                    list_op(
                        [
                            ast_.In(ast_.Identifier(fresh_name), left),
                            right_join_op(ast_.Identifier(fresh_name), right),
                        ],
                    ),
                    ast_.Identifier(fresh_name),
                )
                new_ast._bound_identifiers = {ast_.Identifier(fresh_name)}
                # self.current_bound_identifiers.append(new_ast._bound_identifiers)
                return new_ast

        return None

    def membership_collapse(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.Identifier(_) as x,
                ast_.Quantifier(predicate, expression, op_type),
            ) if op_type.is_collection_operator():

                # If x is not bound by a quantifier, this expression may just be an equality check (ie, is x in {1,2,3}?)
                # rather than a generator
                if x not in self.bound_quantifier_variables:
                    logger.debug(f"FAILED: {x} appears as a generator variable but is not bound by a quantifier")
                    return None

                return ast_.And(
                    [
                        ast_.Equal(x, expression),
                        predicate,
                    ],
                )
        return None


class DisjunctiveNormalFormCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.double_negation,
            self.distribute_de_morgan,
            self.distribute,
            self.flatten_nested_ands,
            self.flatten_nested_ors,
        ]

    def flatten_nested_ands(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.AND):
                if not any(map(lambda x: isinstance(x, ast_.ListOp) and x.op_type == ast_.ListOperator.AND, elems)):
                    return None
                return ast_.And(elems)

        return None

    def flatten_nested_ors(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.OR):
                if not any(map(lambda x: isinstance(x, ast_.ListOp) and x.op_type == ast_.ListOperator.OR, elems)):
                    return None
                return ast_.Or(elems)

        return None

    def double_negation(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.UnaryOp(
                ast_.UnaryOp(
                    x,
                    ast_.UnaryOperator.NOT,
                ),
                ast_.UnaryOperator.NOT,
            ):
                return x
        return None

    def distribute_de_morgan(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.UnaryOp(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                ast_.UnaryOperator.NOT,
            ):
                return ast_.And(
                    [ast_.UnaryOp(elem, ast_.UnaryOperator.NOT) for elem in elems],
                )
            case ast_.UnaryOp(
                ast_.ListOp(elems, ast_.ListOperator.AND),
                ast_.UnaryOperator.NOT,
            ):
                return ast_.Or(
                    [ast_.UnaryOp(elem, ast_.UnaryOperator.NOT) for elem in elems],
                )
        return None

    def distribute(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.AND):
                if not any(map(lambda x: isinstance(x, ast_.ListOp) and x.op_type == ast_.ListOperator.OR, elems)):
                    return None

                or_elems: list[ast_.ListOp] = []
                non_or_elems: list[ast_.ASTNode] = []

                for elem in elems:
                    if isinstance(elem, ast_.ListOp) and elem.op_type == ast_.ListOperator.OR:
                        or_elems.append(elem)
                    else:
                        non_or_elems.append(elem)

                if not or_elems:
                    # Not really a failure, more of a matching refinement
                    logger.debug("FAILED: no OR elements found in AND list operation - this message should be caught beforehand")
                    return None

                new_elems: list[ast_.ASTNode] = []
                or_elem_to_distribute = or_elems[0]
                # Very inefficient, but basically just distribute one or element at a time
                # Calling rewrite rules repeatedly will handle the rest
                non_or_elems = non_or_elems + or_elems[1:]  # type: ignore
                for item in or_elem_to_distribute.items:
                    new_elems.append(ast_.And([item] + non_or_elems))

                return ast_.Or(new_elems)

        return None


class PredicateSimplificationCollection(RewriteCollection):
    def _rewrite_collection(self):
        return [
            self.nesting,
            self.or_wrapping,
        ]

    def nesting(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # TODO, should nesting work with top-level ORs too? should we try to choose nesting more carefully (ie composition chains)
            case ast_.Quantifier(ast_.ListOp(elems, ast_.ListOperator.AND), expression, op_type):
                if not ast._bound_identifiers:
                    logger.debug(f"FAILED: no bound identifiers found in quantifier")
                    return None
                if len(ast._bound_identifiers) <= 1:
                    logger.debug(f"FAILED: not enough bound identifiers to nest quantifier (found {len(ast._bound_identifiers)})")
                    return None

                bound_identifiers = list(ast._bound_identifiers)
                outer_bound_identifier = bound_identifiers[0]
                outer_predicate = list(filter(lambda x: x.contains_item(outer_bound_identifier), elems))
                inner_predicate = list(filter(lambda x: not x.contains_item(outer_bound_identifier), elems))

                inner_quantifier = ast_.Quantifier(
                    ast_.And(inner_predicate),
                    expression,
                    op_type,
                )
                inner_quantifier._bound_identifiers = set(bound_identifiers[1:])

                outer_quantifier = ast_.Quantifier(
                    ast_.And(outer_predicate),
                    inner_quantifier,
                    op_type,
                )
                outer_quantifier._bound_identifiers = set(bound_identifiers[:1])

                return outer_quantifier
        return None

    def or_wrapping(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(ast_.ListOp(_, ast_.ListOperator.AND) as elem, expression, op_type):
                new_quantifier = ast_.Quantifier(
                    ast_.Or([elem]),
                    expression,
                    op_type,
                )
                new_quantifier._bound_identifiers = ast._bound_identifiers
                return new_quantifier

        return None
