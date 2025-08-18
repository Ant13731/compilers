from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection
from src.mod.optimizer.intermediate_ast import (
    GeneratorSelectionV2,
    CombinedGeneratorSelectionV2,
    SingleGeneratorSelectionV2,
    Loop,
)

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


@dataclass
class OrWrappingCollection(RewriteCollection):
    def _rewrite_collection(self):
        return [
            self.or_wrapping,
        ]

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


@dataclass
class GeneratorSelectionCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.generator_selection,
            self.reduce_duplicate_generators,
        ]

    def generator_selection(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                expression,
                op_type,
            ):
                raise NotImplementedError
            # This should only match once per quantifier
            # For each And inside the Or:
            # - Only match if valid generator selection:
            #   - Generator structure exists with LHS in _bound_identifiers
            # - Then:
            #   - Only one generator per bound identifier - need to find all alternatives and then restrict from there
            #   - Try to order in list as a composition chain
        return None

    def reduce_duplicate_generators(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.OR):
                raise NotImplementedError
            # Only match if all elems are either GeneratorSelectionV2 or CombinedGeneratorSelection
            # Check all elements for matches:
            # - elements that share a generator are placed in a combinedgeneratorselection
            # - predicates set to True/None
        return None


@dataclass
class GSPToLoopsCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.summation,
        ]

    def quantifier_generation(
        self,
        ast: ast_.ASTNode,
        op_type: ast_.QuantifierOperator,
        identity: ast_.ASTNode,
        accumulator: Callable[[ast_.ASTNode, ast_.ASTNode], ast_.ASTNode],
        accumulator_type: ast_.SimileType,
    ) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                predicate,  # Should be an OR[GeneratorSelection | CombinedGeneratorSelection]
                expression,
                op_type_,
            ) if (
                op_type_ == op_type
            ):
                if not isinstance(predicate, ast_.ListOp) or predicate.op_type != ast_.ListOperator.OR:
                    logger.debug(f"FAILED: predicate is not a ListOp with OR operator (got {predicate}). This should be in DNF")
                    return None
                if not all(map(lambda x: isinstance(x, GeneratorSelectionV2) or isinstance(x, CombinedGeneratorSelectionV2), predicate.items)):
                    logger.debug(f"FAILED: not all items in predicate are GeneratorSelections (got {predicate.items}). Elements of predicates should be GeneratorSelectionASTs")
                    return None

                assert ast._env is not None, f"Environment should not be None (in quantifier_generation, ast {ast})"
                accumulator_var = ast_.Identifier(self._get_fresh_identifier_name())
                ast._env.put(accumulator_var.name, accumulator_type)

                predicate = ast_.Or(predicate.items)

                if_statement = Loop(
                    predicate,
                    ast_.Assignment(
                        accumulator_var,
                        accumulator(accumulator_var, expression),
                    ),
                )
                return ast_.Statements(
                    [
                        ast_.Assignment(
                            accumulator_var,
                            identity,
                        ),
                        if_statement,
                    ]
                )
        return None

    def summation(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        return self.quantifier_generation(
            ast,
            ast_.QuantifierOperator.SUM,
            ast_.Int("0"),
            ast_.Add,
            ast_.BaseSimileType.Int,
        )

    def top_level_or_loop(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                ast_.ListOp(predicates, ast_.ListOperator.OR),
                body,
            ):
                ...
        return None

    def chained_gsp_loop(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                GeneratorSelectionV2(generators, predicates),
                body,
            ):
                ...
        return None

    def single_gsp_loop(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                GeneratorSelectionV2([generator], predicates),
                body,
            ):
                ...
        return None

    def combined_gsp_loop(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                CombinedGeneratorSelectionV2(
                    generator,
                    gsp_predicates,
                    predicates,
                ),
                body,
            ):
                ...
        return None


class LoopsCodeGenerationCollection(RewriteCollection):

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.conjunct_conditional,
        ]

    def conjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                SingleGeneratorSelectionV2(
                    generator,
                    predicates,
                ),
                body,
            ):
                ...
        return None


class ReplaceAndSimplifyCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.equality_elimination,
            self.simplify_equalities,
            self.simplify_and_true,
            self.simplify_and_false,
            self.simplify_or_true,
            self.simplify_or_false,
            self.flatten_nested_statements,
        ]
