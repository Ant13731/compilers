from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection
from src.mod.optimizer.intermediate_ast import GeneratorSelectionAST

# NOTE: REWRITE RULES MUST ALWAYS USE THE PARENT FORM FOR STRUCTURAL MATCHING (ex. BinaryOp instead of Add)


# Rewrite rules written in plain form:
# predicate_operations:
#   S ∪ T ~> {x | x ∈ S ∨ x ∈ T}
#   S ∩ T ~> {x | x ∈ S ∧ x ∈ T}
#   S \ T ~> {x | x ∈ S ∧ x ∉ T}
# singleton_membership:
#   x ∈ {y} ~> x = y
# membership_collapse:
#   ⊕_q f(x) | x ∈ {g(y) | y ∈ S} ∧ p(x)                       ~> ⊕_q f(g(y)) | y ∈ S ∧ p(g(y))
#               x ∈ {g(y) | y ∈ S} ∧ p(x) ∨ x ∈ {h(z) | z ∈ T} ~> ⊕_q f(x) | y ∈ S ∧ x = g(y) ∧ p(x) ∨ z ∈ T ∧ x = h(z)
# new_membership_collapse: without the context of a quantifier
#   x ∈ {g(y) | y ∈ S ∧ P ∨ Q} ∧ F ~> x = g(y) ∧ y ∈ S ∧ P ∨ Q ∧ F
# bubble_up_generators:
#   (x = g(y) ∧ y ∈ S ∧ A) ∨ (x = h(z) ∧ x ∈ T ∧ B) ∧ C


@dataclass
class SetComprehensionConstructionCollection(RewriteCollection):

    bound_quantifier_variables: set[ast_.Identifier] = field(default_factory=set)
    current_bound_identifiers: list[set[ast_.Identifier]] = field(default_factory=list)

    def apply_all_rules_one_traversal(self, ast):
        bound_quantifier_variables_before = deepcopy(self.bound_quantifier_variables)
        if isinstance(ast, ast_.Quantifier):
            logger.debug(f"Quantifier found with bound variables: {ast.bound}")
            self.bound_quantifier_variables |= ast.bound
            self.current_bound_identifiers.append(ast._bound_identifiers)

        ast = super().apply_all_rules_one_traversal(ast)

        self.bound_quantifier_variables = bound_quantifier_variables_before
        if self.current_bound_identifiers and hasattr(ast, "_bound_identifiers") and ast._bound_identifiers != self.current_bound_identifiers:
            # If the current bound identifiers are set, we need to update the AST's bound identifiers
            ast._bound_identifiers = self.current_bound_identifiers.pop()
        return ast

    # Collection setTypes should have no "demoted" predicates yet
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.predicate_operations_union,
            self.predicate_operations_intersection,
            self.predicate_operations_difference,
            # self.singleton_membership,
            self.membership_collapse,
        ]

    def predicate_operations_union(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # Idea, if we want to add some compilation-time optimizations (like union of two set enums into one set enum),
            # we can just add those rules here
            case ast_.BinaryOp(left, right, ast_.BinaryOperator.UNION):
                if any(
                    [
                        not isinstance(left.get_type, ast_.SetType),
                        not isinstance(right.get_type, ast_.SetType),
                    ]
                ):
                    logger.debug(f"FAILED: at least one union child is not a set type: {left.get_type}, {right.get_type}")
                    return None
                fresh_name = self._get_fresh_identifier_name()
                new_ast = ast_.SetComprehension(
                    ast_.ListOp.flatten_and_join(
                        [
                            ast_.In(ast_.Identifier(fresh_name), left),
                            ast_.In(ast_.Identifier(fresh_name), right),
                        ],
                        ast_.ListOperator.OR,
                    ),
                    ast_.Identifier(fresh_name),
                )
                new_ast._bound_identifiers = {ast_.Identifier(fresh_name)}
                return analysis.add_environments_to_ast(new_ast, ast._env)

        return None

    def predicate_operations_intersection(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(left, right, ast_.BinaryOperator.INTERSECTION):
                if any(
                    [
                        not isinstance(left.get_type, ast_.SetType),
                        not isinstance(right.get_type, ast_.SetType),
                    ]
                ):
                    logger.debug(f"FAILED: at least one intersection child is not a set type: {left.get_type}, {right.get_type}")
                    return None
                fresh_name = self._get_fresh_identifier_name()
                new_ast = ast_.SetComprehension(
                    ast_.ListOp.flatten_and_join(
                        [
                            ast_.In(ast_.Identifier(fresh_name), left),
                            ast_.In(ast_.Identifier(fresh_name), right),
                        ],
                        ast_.ListOperator.AND,
                    ),
                    ast_.Identifier(fresh_name),
                )
                new_ast._bound_identifiers = {ast_.Identifier(fresh_name)}
                return analysis.add_environments_to_ast(new_ast, ast._env)

        return None

    def predicate_operations_difference(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(left, right, ast_.BinaryOperator.DIFFERENCE):
                if any(
                    [
                        not isinstance(left.get_type, ast_.SetType),
                        not isinstance(right.get_type, ast_.SetType),
                    ]
                ):
                    logger.debug(f"FAILED: at least one difference child is not a set type: {left.get_type}, {right.get_type}")
                    return None
                fresh_name = self._get_fresh_identifier_name()
                new_ast = ast_.SetComprehension(
                    ast_.ListOp.flatten_and_join(
                        [
                            ast_.In(ast_.Identifier(fresh_name), left),
                            ast_.NotIn(ast_.Identifier(fresh_name), right),
                        ],
                        ast_.ListOperator.AND,
                    ),
                    ast_.Identifier(fresh_name),
                )
                new_ast._bound_identifiers = {ast_.Identifier(fresh_name)}
                return analysis.add_environments_to_ast(new_ast, ast._env)

        return None

    # def singleton_membership(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.BinaryOp(ast_.Identifier(_) as x, ast_.Enumeration([elem], _), ast_.BinaryOperator.IN):
    #             new_ast = ast_.Equal(x, elem)
    #             return analysis.add_environments_to_ast(new_ast, ast._env)

    #     return None

    def membership_collapse(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.Identifier(_) as x,
                ast_.Quantifier(
                    predicate,
                    expression,
                    op_type,
                ) as inner_quantifier,
            ) if op_type.is_collection_operator():

                # If x is not bound by a quantifier, this expression may just be an equality check (ie, is 1 in {1,2,3}?)
                # rather than a generator
                if x not in self.bound_quantifier_variables:
                    logger.debug(f"FAILED: {x} appears as a generator variable but is not bound by a quantifier")
                    return None

                if inner_quantifier._bound_identifiers is not None:
                    self.current_bound_identifiers[-1].update(inner_quantifier._bound_identifiers)

                return analysis.add_environments_to_ast(
                    ast_.ListOp.flatten_and_join(
                        [
                            ast_.Equal(x, expression),
                            predicate,
                        ],
                        ast_.ListOperator.AND,
                    ),
                    ast._env,
                )
        return None


class DisjunctiveNormalFormQuantifierPredicateCollection(RewriteCollection):
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
                return analysis.add_environments_to_ast(
                    ast_.ListOp.flatten_and_join(elems, ast_.ListOperator.AND),
                    ast._env,
                )
        return None

    def flatten_nested_ors(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.OR):
                if not any(map(lambda x: isinstance(x, ast_.ListOp) and x.op_type == ast_.ListOperator.OR, elems)):
                    return None
                return analysis.add_environments_to_ast(
                    ast_.ListOp.flatten_and_join(elems, ast_.ListOperator.OR),
                    ast._env,
                )
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
                return analysis.add_environments_to_ast(x, ast._env)
        return None

    def distribute_de_morgan(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.UnaryOp(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                ast_.UnaryOperator.NOT,
            ):
                return analysis.add_environments_to_ast(
                    ast_.ListOp(
                        [ast_.UnaryOp(elem, ast_.UnaryOperator.NOT) for elem in elems],
                        ast_.ListOperator.AND,
                    ),
                    ast._env,
                )
            case ast_.UnaryOp(
                ast_.ListOp(elems, ast_.ListOperator.AND),
                ast_.UnaryOperator.NOT,
            ):
                return analysis.add_environments_to_ast(
                    ast_.ListOp(
                        [ast_.UnaryOp(elem, ast_.UnaryOperator.NOT) for elem in elems],
                        ast_.ListOperator.OR,
                    ),
                    ast._env,
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
                    new_elems.append(
                        ast_.And(
                            [item] + non_or_elems,
                        )
                    )

                return analysis.add_environments_to_ast(ast_.Or(new_elems), ast._env)

        return None


class PredicateSimplificationCollection(RewriteCollection):
    def _rewrite_collection(self):
        return [
            self.nesting,
            self.or_wrapping,
        ]

    def nesting(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
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

                return analysis.add_environments_to_ast(outer_quantifier, ast._env)
        return None

    def or_wrapping(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(ast_.ListOp(_, ast_.ListOperator.AND) as elem, expression, op_type):
                new_quantifier = ast_.Quantifier(
                    ast_.ListOp([elem], ast_.ListOperator.OR),
                    expression,
                    op_type,
                )
                new_quantifier._bound_identifiers = ast._bound_identifiers
                return analysis.add_environments_to_ast(new_quantifier, ast._env)

        return None


class GeneratorSelectionCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            # self.select_generator_predicates_or,
            # self.select_generator_predicates_and,
            # self.set_generation,
            self.generator_selection_and_dummy_reassignment,
            self.reduce_duplicate_generators,
        ]

    def generator_selection_and_dummy_reassignment(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                expression,
                op_type,
            ):
                if all(map(lambda x: isinstance(x, GeneratorSelectionAST), elems)):
                    logger.debug(f"FAILED: all elements in OR quantifier are already GeneratorSelectionASTs, no need to select generators")
                    return None

                predicates: list[ast_.ASTNode] = []
                for elem in elems:
                    if isinstance(elem, ast_.BinaryOp) and elem.op_type == ast_.BinaryOperator.IN:
                        predicates.append(
                            GeneratorSelectionAST(
                                generator=elem,
                                assignments=[],
                                condition=ast_.And([]),
                            )
                        )
                        continue

                    if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
                        logger.debug(
                            f"FAILED: element {elem} is not a ListOp with AND operator (and not a generator itself). Expected predicates to be in disjunctive normal form (an OR of ANDs)"
                        )
                        return None

                    candidate_generators = []
                    equality_assignments = []
                    other_predicates = []
                    for item in elem.items:
                        match item:
                            case ast_.BinaryOp(
                                ast_.Identifier(_) as x,
                                set_type,
                                ast_.BinaryOperator.IN,
                            ) if isinstance(
                                set_type.get_type, ast_.SetType
                            ) and (ast.bound is None or x in ast.bound):
                                candidate_generators.append(item)
                            case ast_.BinaryOp(
                                ast_.Identifier(_) as left,
                                right,
                                eq_type,
                            ) if all(
                                [
                                    eq_type == ast_.BinaryOperator.EQUAL,
                                    left in ast.bound,
                                ]
                            ):
                                equality_assignments.append(ast_.Assignment(left, right))
                            case ast_.BinaryOp(
                                left,
                                ast_.Identifier(_) as right,
                                eq_type,
                            ) if all(
                                [
                                    eq_type == ast_.BinaryOperator.EQUAL,
                                    left in ast.bound,
                                ]
                            ):
                                equality_assignments.append(ast_.Assignment(right, left))
                            case _:
                                other_predicates.append(item)
                    if not candidate_generators:
                        logger.debug(f"FAILED: no candidate generators found in AND predicate (bound variables are {ast.bound}, free are {ast.free})")
                        return None

                    selected_generator = candidate_generators[0]
                    other_predicates += candidate_generators[1:]

                    predicates.append(
                        GeneratorSelectionAST(
                            generator=selected_generator,
                            assignments=equality_assignments,
                            condition=ast_.And.flatten_and_join(other_predicates, ast_.ListOperator.AND),
                        )
                    )

                return analysis.add_environments_to_ast(
                    ast_.Quantifier(
                        ast_.Or(predicates),
                        expression,
                        op_type,
                    ),
                    ast._env,
                )
        return None

    def reduce_duplicate_generators(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                expression,
                op_type,
            ) if all(map(lambda x: isinstance(x, GeneratorSelectionAST), elems)):

                reduced_elem = False
                condensed_elems: list[ast_.ASTNode] = []
                for i in range(len(elems)):
                    elem = elems[i]
                    assert isinstance(elem, GeneratorSelectionAST)
                    for j in range(i + 1, len(elems)):
                        other_elem = elems[j]
                        assert isinstance(other_elem, GeneratorSelectionAST)

                        if elem.generator == other_elem.generator:
                            new_elem = GeneratorSelectionAST(
                                generator=elem.generator,
                                assignments=elem.assignments + other_elem.assignments,
                                condition=ast_.And.flatten_and_join([elem.condition, other_elem.condition], ast_.ListOperator.AND),
                            )
                            condensed_elems.append(new_elem)
                            reduced_elem = True
                            break
                    else:
                        condensed_elems.append(elem)

                if not reduced_elem:
                    logger.debug(f"FAILED: no duplicate generators found in OR quantifier (bound variables are {ast.bound}, free are {ast.free})")
                    return None

                return analysis.add_environments_to_ast(
                    ast_.Quantifier(
                        ast_.ListOp(condensed_elems, ast_.ListOperator.OR),
                        expression,
                        op_type,
                    ),
                    ast._env,
                )
        return None

    # def select_generator_predicates_or(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(ast_.ListOp(elems, ast_.ListOperator.OR), _, _):
    #             if ast._selected_generators is not None:
    #                 return None

    #             generators = []
    #             for elem in elems:
    #                 if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
    #                     continue

    #                 candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.bound)
    #                 if not candidate_generators:
    #                     logger.debug(f"FAILED: no candidate generators found in OR predicate (bound variables are {ast.bound}, free are {ast.free})")
    #                     return None

    #                 # TODO For now, just manually choose the first available generator
    #                 generators.append(candidate_generators[0])

    #             ast._selected_generators = generators
    #             return ast
    #     return None

    # def select_generator_predicates_and(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(ast_.ListOp(_, ast_.ListOperator.AND) as elem, _, _):
    #             if ast._selected_generators is not None:
    #                 return None

    #             candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.bound)
    #             if not candidate_generators:
    #                 logger.debug(f"FAILED: no candidate generators found in AND predicate (bound variables are {ast.bound}, free are {ast.free})")
    #                 return None

    #             ast._selected_generators = [candidate_generators[0]]
    #             return ast
    #     return None


class SetCodeGenerationCollection(RewriteCollection):

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.summation,
            self.flatten_nested_statements,
            self.disjunct_conditional,
            self.conjunct_conditional,
            # self.set_generation,
        ]

    def flatten_nested_statements(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Statements(items):

                if not any(map(lambda x: isinstance(x, ast_.Statements), items)):
                    return None

                # new_env =
                # assert isinstance(new_env, ast_.Environment), f"Environment should not be None (in ast {ast})"

                new_statements: list[ast_.ASTNode] = []
                for item in items:
                    if not isinstance(item, ast_.Statements):
                        new_statements.append(item)
                        continue

                    new_statements.extend(item.items)
                    # assert isinstance(item._env, ast_.Environment), f"Environment should not be None (in item {item})"
                    # for key, value in item._env.table.items():
                    #     if key not in new_env.table:
                    #         new_env.put(key, value)

                return analysis.add_environments_to_ast(ast_.Statements(new_statements), ast._env)
        return None

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
                predicate,  # Should be an OR[GeneratorSelectionAST]s
                expression,
                op_type_,
            ) if (
                op_type_ == op_type
            ):
                if not isinstance(predicate, ast_.ListOp) or predicate.op_type != ast_.ListOperator.OR:
                    logger.debug(f"FAILED: predicate is not a ListOp with OR operator (got {predicate}). This should be in DNF")
                    return None
                if not all(map(lambda x: isinstance(x, GeneratorSelectionAST), predicate.items)):
                    logger.debug(f"FAILED: not all items in predicate are GeneratorSelectionASTs (got {predicate.items}). Elements of predicates should be GeneratorSelectionASTs")
                    return None

                assert ast._env is not None, f"Environment should not be None (in quantifier_generation, ast {ast})"
                accumulator_var = ast_.Identifier(self._get_fresh_identifier_name())
                ast._env.put(accumulator_var.name, accumulator_type)

                if_statement = ast_.If(
                    predicate,
                    ast_.Assignment(
                        accumulator_var,
                        accumulator(accumulator_var, expression),
                    ),
                )
                return analysis.add_environments_to_ast(
                    ast_.Statements(
                        [
                            ast_.Assignment(
                                accumulator_var,
                                identity,
                            ),
                            if_statement,
                        ]
                    ),
                    ast._env,
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
        # match ast:
        #     case ast_.Quantifier(
        #         predicate,
        #         expression,
        #         ast_.QuantifierOperator.SUM,
        #     ):
        #         # Idea - put everything in an if statement to start
        #         counter = ast_.Identifier(self._get_fresh_identifier_name())

        #         assert ast._env is not None, f"Environment should not be None (in summation, ast {ast})"
        #         ast._env.put(counter.name, ast_.BaseSimileType.Int)

        #         if_statement = ast_.If(
        #             predicate,
        #             ast_.Assignment(
        #                 counter,
        #                 ast_.Add(
        #                     counter,
        #                     expression,
        #                 ),
        #             ),
        #         )
        #         if_statement._rewrite_generators = ast._selected_generators
        #         if_statement._bound_by_quantifier_rewrite = ast.bound

        #         return analysis.add_environments_to_ast(
        #             ast_.Statements(
        #                 [
        #                     ast_.Assignment(
        #                         counter,
        #                         ast_.Int("0"),
        #                     ),
        #                     if_statement,
        #                 ]
        #             ),
        #             ast._env,
        #         )
        # new_statement._env = ast._env
        # assert new_statement._env is not None, f"Environment should not be None (in summation, ast {new_statement})"
        # new_statement._env.put(counter.name, ast_.BaseSimileType.Int)
        # return new_statement

        # return None

    def disjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.If(
                ast_.ListOp(predicates, ast_.ListOperator.OR),
                body,
            ):
                statements: list[ast_.ASTNode] = []
                past_predicates: list[ast_.ASTNode] = []

                # Every predicate should have a generator matching 1-1 from the GeneratorSelectionCollection
                for i, predicate in enumerate(predicates):
                    if not isinstance(predicate, GeneratorSelectionAST):
                        logger.debug(f"FAILED: predicate {predicate} is not a GeneratorSelectionAST but should be")
                        return None

                    past_predicate_inverse: ast_.Not | None = None
                    if past_predicates:
                        past_predicate_inverse = ast_.Not(
                            ast_.ListOp.flatten_and_join(
                                past_predicates,
                                ast_.ListOperator.AND,
                            ),
                        )

                    if_statement = ast_.If(
                        predicate.with_new_conditions(past_predicate_inverse),
                        body,
                    )
                    # if_statement._rewrite_generators = [ast._rewrite_generators[i]]
                    # if_statement._bound_by_quantifier_rewrite = ast._bound_by_quantifier_rewrite
                    statements.append(if_statement)
                    past_predicates.append(predicate.flatten())

                return analysis.add_environments_to_ast(
                    ast_.Statements(
                        statements,
                    ),
                    ast._env,
                )
        return None
        # match ast:
        #     case ast_.If(
        #         ast_.ListOp(predicates, ast_.ListOperator.OR),
        #         body,
        #     ):
        #         if ast._rewrite_generators is None:
        #             logger.debug(f"FAILED: no candidate generators found in disjunctive predicate")
        #             return None
        #         if ast._bound_by_quantifier_rewrite is None:
        #             logger.debug(f"FAILED: no bound variables found in disjunctive predicate")
        #             return None

        #         statements: list[ast_.ASTNode] = []
        #         past_predicates: list[ast_.ASTNode] = []

        #         assert len(ast._rewrite_generators) == len(
        #             predicates
        #         ), f"Every OR predicate should have a generator, but got {len(ast._rewrite_generators)} generators and {len(predicates)} predicates"

        #         # Every predicate should have a generator matching 1-1 from the GeneratorSelectionCollection
        #         for i, predicate in enumerate(predicates):
        #             past_predicate_inverse = []
        #             if past_predicates:
        #                 past_predicate_inverse = [
        #                     ast_.Not(
        #                         ast_.ListOp.flatten_and_join(
        #                             past_predicates,
        #                             ast_.ListOperator.AND,
        #                         ),
        #                     )
        #                 ]

        #             if_statement = ast_.If(
        #                 ast_.ListOp.flatten_and_join(
        #                     [predicate] + past_predicate_inverse,
        #                     ast_.ListOperator.AND,
        #                 ),
        #                 body,
        #             )
        #             if_statement._rewrite_generators = [ast._rewrite_generators[i]]
        #             if_statement._bound_by_quantifier_rewrite = ast._bound_by_quantifier_rewrite
        #             statements.append(if_statement)
        #             past_predicates.append(predicate)
        #         return analysis.add_environments_to_ast(
        #             ast_.Statements(
        #                 statements,
        #             ),
        #             ast._env,
        #         )
        # return None

    def conjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.If(
                GeneratorSelectionAST(generator, assignments, condition),
                body,
            ):
                if not isinstance(generator, ast_.BinaryOp) or generator.op_type != ast_.BinaryOperator.IN or not isinstance(generator.left, ast_.Identifier):
                    logger.debug(f"FAILED: generator is not a valid IN operation (got {generator})")
                    return None

                free_predicates = []
                bound_predicates = []
                for condition_item in condition.items:
                    if condition_item.contains_item(generator.left):
                        bound_predicates.append(condition_item)
                        continue
                    for bound_var in ast.bound:
                        if bound_var in condition_item.free:
                            bound_predicates.append(condition_item)
                            break
                    else:
                        # If no bound variable is found in the condition item, it is a free predicate
                        free_predicates.append(condition_item)

                # free_predicates = list(filter(lambda x: not x.contains_item(generator.left), condition.items))
                # bound_predicates = list(filter(lambda x: x.contains_item(generator.left), condition.items))

                statement: ast_.ASTNode = body
                if bound_predicates:
                    statement = ast_.Statements(
                        [
                            ast_.If(
                                ast_.ListOp.flatten_and_join(
                                    bound_predicates,
                                    ast_.ListOperator.AND,
                                ),
                                statement,
                            )
                        ]
                    )

                if assignments:
                    statement = ast_.Statements(
                        assignments + [statement],
                    )

                statement = ast_.For(
                    ast_.IdentList([generator.left]),
                    generator.right,
                    statement,
                )

                if free_predicates:
                    statement = ast_.If(
                        ast_.ListOp.flatten_and_join(
                            free_predicates,
                            ast_.ListOperator.AND,
                        ),
                        body=ast_.Statements([statement]),
                    )

                return analysis.add_environments_to_ast(
                    ast_.Statements([statement]),
                    ast._env,
                )
        return None
        #         if ast._rewrite_generators is None:
        #             logger.debug(f"FAILED: no candidate generators found in conjunctive predicate")
        #             return None
        #         if ast._bound_by_quantifier_rewrite is None:
        #             logger.debug(f"FAILED: no bound variables found in conjunctive predicate")
        #             return None

        #         if not ast._rewrite_generators:
        #             logger.debug(f"FAILED: no candidate generators found in conjunctive predicate")
        #             return None
        #         if len(ast._rewrite_generators) != 1:
        #             logger.debug(f"FAILED: more than one candidate generator found in conjunctive predicate: {ast._rewrite_generators}")
        #             return None

        #         generator = ast._rewrite_generators[0]
        #         if not isinstance(generator, ast_.BinaryOp) or generator.op_type != ast_.BinaryOperator.IN or not isinstance(generator.left, ast_.Identifier):
        #             logger.debug(f"FAILED: generator is not a valid IN operation (got {generator})")
        #             return None

        #         predicates = list(filter(lambda x: x != generator, predicates_))
        #         free_predicates = list(filter(lambda x: not x.contains_item(generator.left), predicates))
        #         bound_predicates = list(filter(lambda x: x.contains_item(generator.left), predicates))

        #         statement: ast_.ASTNode = body
        #         if bound_predicates:
        #             # Handle bound-to-bound variable equality checks as assignments
        #             if_statement_bound_predicates = []

        #             assignment_statements: list[ast_.ASTNode] = []
        #             for bound_predicate in bound_predicates:
        #                 if not isinstance(bound_predicate, ast_.BinaryOp) or not bound_predicate.op_type == ast_.BinaryOperator.EQUAL:
        #                     if_statement_bound_predicates.append(bound_predicate)
        #                     continue

        #                 for bound_var in ast._bound_by_quantifier_rewrite:
        #                     if bound_var not in bound_predicate.left.free and generator.left not in bound_predicate.right.free:
        #                         continue

        #                     assignment_statements.append(
        #                         ast_.Assignment(
        #                             bound_predicate.left,
        #                             bound_predicate.right,  # The right side should have the generator variable
        #                         ),
        #                     )
        #                     break

        #             if if_statement_bound_predicates:
        #                 statement = ast_.Statements(
        #                     assignment_statements
        #                     + [
        #                         ast_.If(
        #                             ast_.ListOp.flatten_and_join(
        #                                 if_statement_bound_predicates,
        #                                 ast_.ListOperator.AND,
        #                             ),
        #                             statement,
        #                         )
        #                     ]
        #                 )
        #             else:
        #                 statement = ast_.Statements(
        #                     assignment_statements + [statement],
        #                 )

        #         statement = ast_.For(
        #             ast_.IdentList([generator.left]),
        #             generator.right,
        #             statement,
        #         )

        #         if free_predicates:
        #             statement = ast_.If(
        #                 ast_.ListOp.flatten_and_join(
        #                     free_predicates,
        #                     ast_.ListOperator.AND,
        #                 ),
        #                 body=ast_.Statements([statement]),
        #             )

        #         return analysis.add_environments_to_ast(
        #             ast_.Statements([statement]),
        #             ast._env,
        #         )
        # return None

    # c = 0
    # if P:
    #     c += E

    # if P is not a top-level OR (ie. is a top-level AND)
    # c = 0
    # if P.free_predicate (where P is free):
    #   # Also replace equalities with top-level bound variable - ie predicate x | x = f(y) and y in S should become f(y) | y in S. Can likely use assignment before the if statement
    #   for a in P.generator:
    #       if P.bound_predicate:
    #           c += E

    # if P is a top-level OR
    # c = 0
    # if P.clause[0]:
    #   c += E
    # if P.clause[1] and not P.clause[0]:
    #   c += E
    # if P.clause[2] and not (P.clause[0] or P.clause[1]):
    #   c += E
    # ...

    # def composed_contitional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.If(
    #             ast_.ListOp(predicates, ast_.ListOperator.AND),
    #             body,
    #         ):
    #             pass

    #     return None

    # def summation(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, ast_.ListOperator.AND) as predicate,
    #             expression,
    #             ast_.QuantifierOperator.SUM,
    #         ):
    #             candidate_generators, predicates = predicate.separate_candidate_generators_from_predicates(ast.bound)
    #             if not candidate_generators:
    #                 logger.debug(f"FAILED: no candidate generators found in summation predicate (bound variables are {ast.bound}, free are {ast.free})")
    #                 return None

    #             # TODO For now, just manually choose the first available generator
    #             # Maybe optimize based on set size if sets are enumerated at compile time?
    #             candidate_generator = None
    #             for candidate in candidate_generators:
    #                 if candidate.op_type != ast_.BinaryOperator.IN:
    #                     continue
    #                 if not isinstance(candidate.left, ast_.Identifier):
    #                     continue
    #                 candidate_generator = candidate
    #                 candidate_generators.pop(candidate_generators.index(candidate))
    #                 break
    #             else:
    #                 logger.debug(f"FAILED: no candidate generator with proper LH identifier found in summation predicate. Got {candidate_generators}")
    #                 return None
    #             predicates = predicates + candidate_generators

    #             counter = ast_.Identifier(self._get_fresh_identifier_name())
    #             new_statement = ast_.Statements(
    #                 [
    #                     ast_.Assignment(
    #                         counter,
    #                         ast_.Int("0"),
    #                     ),
    #                     ast_.For(
    #                         candidate_generator.left,
    #                         candidate_generator.right,
    #                         ast_.Statements(
    #                             [
    #                                 ast_.If(
    #                                     ast_.And(predicates),
    #                                     ast_.Assignment(
    #                                         counter,
    #                                         ast_.Add(
    #                                             counter,
    #                                             expression,
    #                                         ),
    #                                     ),
    #                                 )
    #                             ],
    #                         ),
    #                     ),
    #                 ]
    #             )
    #             new_statement._env = ast._env
    #             assert new_statement._env is not None, f"Environment should not be None (in summation, ast {new_statement})"
    #             return new_statement
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, ast_.ListOperator.OR) as predicate,
    #             expression,
    #             ast_.QuantifierOperator.SUM,
    #         ):
    #             counter = ast_.Identifier(self._get_fresh_identifier_name())
    #             statements: list[ast_.ASTNode] = [
    #                 ast_.Assignment(
    #                     counter,
    #                     ast_.Int("0"),
    #                 )
    #             ]

    #             prev_predicates: list[ast_.ASTNode] = []
    #             for elem in elems:
    #                 if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
    #                     logger.debug(f"FAILED: {elem} is not a valid AND list operation (predicate should be in disjunctive normal form)")
    #                     return None
    #                 candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.bound)

    #                 candidate_generator = None
    #                 for candidate in candidate_generators:
    #                     if candidate.op_type != ast_.BinaryOperator.IN:
    #                         continue
    #                     if not isinstance(candidate.left, ast_.Identifier):
    #                         continue
    #                     candidate_generator = candidate
    #                     candidate_generators.pop(candidate_generators.index(candidate))
    #                     break
    #                 else:
    #                     logger.debug(f"FAILED: no candidate generator with proper LH identifier found in summation predicate. Got {candidate_generators}")
    #                     return None

    #                 predicates = predicates + candidate_generators
    #                 statements.append(
    #                     ast_.For(
    #                         candidate_generator.left,
    #                         candidate_generator.right,
    #                         ast_.Statements(
    #                             [
    #                                 ast_.If(
    #                                     ast_.And(predicates + prev_predicates),
    #                                     ast_.Assignment(
    #                                         counter,
    #                                         ast_.Add(
    #                                             counter,
    #                                             expression,
    #                                         ),
    #                                     ),
    #                                 )
    #                             ],
    #                         ),
    #                     )
    #                 )
    #                 prev_predicates.append(ast_.Not(ast_.And(predicates)))

    #             new_statement = ast_.Statements(statements)
    #             new_statement._env = ast._env
    #             return new_statement

    #     return None

    # def set_generation(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, ast_.ListOperator.OR) as predicate,
    #             expression,
    #             op_type,
    #         ):
    #             ret_id = ast_.Identifier(self._get_fresh_identifier_name())

    #             statements: list[ast_.ASTNode] = [
    #                 ast_.Assignment(
    #                     ret_id,
    #                     ast_.SetEnumeration(
    #                         [],
    #                         op_type=ast_.CollectionOperator.SET,
    #                     ),
    #                 ),
    #             ]

    #             for elem in elems:
    #                 if isinstance(elem, ast_.ListOp) and elem.op_type == ast_.ListOperator.AND:
    #                     candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.free)
    #                 elif isinstance(elem, ast_.BinaryOp) and elem.op_type == ast_.BinaryOperator.IN:
    #                     candidate_generators = [elem]
    #                     predicates = []
    #                 else:
    #                     return None

    #                 if not candidate_generators:
    #                     return None

    #                 candidate_generator = next(iter(candidate_generators))
    #                 if candidate_generator.op_type != ast_.BinaryOperator.IN:
    #                     return None
    #                 if not isinstance(candidate_generator.left, ast_.Identifier):
    #                     return None

    #                 statements.append(
    #                     ast_.For(
    #                         ast_.IdentList([candidate_generator.left]),
    #                         candidate_generator.right,
    #                         ast_.Statements(
    #                             [
    #                                 ast_.If(
    #                                     ast_.And(predicates + candidate_generators[1:]),
    #                                     ast_.Call(
    #                                         ast_.StructAccess(
    #                                             ret_id,
    #                                             ast_.Identifier("add"),
    #                                         ),
    #                                         [expression],
    #                                     ),
    #                                     ast_.None_(),
    #                                 )
    #                             ]
    #                         ),
    #                     )
    #                 )
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, ast_.ListOperator.AND) as predicate,
    #             expression,
    #             op_type,
    #         ):
    #             ret_id = ast_.Identifier(self._get_fresh_identifier_name())

    #             statements = [
    #                 ast_.Assignment(
    #                     ret_id,
    #                     ast_.SetEnumeration(
    #                         [],
    #                         op_type=ast_.CollectionOperator.SET,
    #                     ),
    #                 ),
    #             ]

    #             candidate_generators, predicates = predicate.separate_candidate_generators_from_predicates(ast.free)

    #             if not candidate_generators:
    #                 return None

    #             candidate_generator = next(iter(candidate_generators))
    #             if candidate_generator.op_type != ast_.BinaryOperator.IN:
    #                 return None
    #             if not isinstance(candidate_generator.left, ast_.Identifier):
    #                 return None

    #             statements.append(
    #                 ast_.For(
    #                     ast_.IdentList([candidate_generator.left]),
    #                     candidate_generator.right,
    #                     ast_.Statements(
    #                         [
    #                             ast_.If(
    #                                 ast_.And(predicates + candidate_generators[1:]),
    #                                 ast_.Call(
    #                                     ast_.StructAccess(
    #                                         ret_id,
    #                                         ast_.Identifier("add"),
    #                                     ),
    #                                     [expression],
    #                                 ),
    #                                 ast_.None_(),
    #                             )
    #                         ]
    #                     ),
    #                 )
    #             )

    #     return None

    # def summation(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    # def composed_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    # def conjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:


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
# TEST_PHASE = SetRewriteCollection()
# print(TEST_PHASE.normalize(TEST_TYPE).pretty_print(print_env=False))

SET_REWRITE_COLLECTION: list[type[RewriteCollection]] = [
    SetComprehensionConstructionCollection,
    DisjunctiveNormalFormQuantifierPredicateCollection,
    PredicateSimplificationCollection,
    GeneratorSelectionCollection,
    SetCodeGenerationCollection,
]
