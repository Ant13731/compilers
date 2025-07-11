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
        logger.debug(f"AST after applying all rules: {ast.pretty_print_algorithmic()}")

        self.bound_quantifier_variables = bound_quantifier_variables_before
        if self.current_bound_identifiers and hasattr(ast, "_bound_identifiers") and ast._bound_identifiers != self.current_bound_identifiers:
            # If the current bound identifiers are set, we need to update the AST's bound identifiers
            ast._bound_identifiers = self.current_bound_identifiers.pop()
        logger.debug(f"AST after swapping identifiers: {ast.pretty_print_algorithmic()}")

        return ast

    # Collection setTypes should have no "demoted" predicates yet
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
                        list_op_type = ast_.ListOperator.OR
                        right = ast_.In(ast_.Identifier(fresh_name), right)
                    case ast_.BinaryOperator.INTERSECTION:
                        list_op_type = ast_.ListOperator.AND
                        right = ast_.In(ast_.Identifier(fresh_name), right)
                    case ast_.BinaryOperator.DIFFERENCE:
                        list_op_type = ast_.ListOperator.AND
                        right = ast_.NotIn(ast_.Identifier(fresh_name), right)

                new_ast = ast_.SetComprehension(
                    ast_.ListOp.flatten_and_join(
                        [
                            ast_.In(ast_.Identifier(fresh_name), left),
                            right,
                        ],
                        list_op_type,
                    ),
                    ast_.Identifier(fresh_name),
                )
                new_ast._bound_identifiers = {ast_.Identifier(fresh_name)}
                self.current_bound_identifiers.append(new_ast._bound_identifiers)
                return new_ast

        return None

    # def singleton_membership(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.BinaryOp(ast_.Identifier(_) as x, ast_.Enumeration([elem], _), ast_.BinaryOperator.IN):
    #             new_ast = ast_.Equal(x, elem)
    #             return new_ast, ast._env)

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

                # Trying to leave inner pred bound vars as free
                # if inner_quantifier._bound_identifiers is not None:
                # logger.debug(f"DEBUG: AST {ast.pretty_print_algorithmic()}")
                # self.current_hidden_bound_identifiers[-1].update(inner_quantifier._bound_identifiers)

                return ast_.ListOp.flatten_and_join(
                    [
                        ast_.Equal(x, expression),
                        predicate,
                    ],
                    ast_.ListOperator.AND,
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
                return ast_.ListOp.flatten_and_join(elems, ast_.ListOperator.AND)

        return None

    def flatten_nested_ors(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.OR):
                if not any(map(lambda x: isinstance(x, ast_.ListOp) and x.op_type == ast_.ListOperator.OR, elems)):
                    return None
                return ast_.ListOp.flatten_and_join(elems, ast_.ListOperator.OR)

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
                return ast_.ListOp(
                    [ast_.UnaryOp(elem, ast_.UnaryOperator.NOT) for elem in elems],
                    ast_.ListOperator.AND,
                )
            case ast_.UnaryOp(
                ast_.ListOp(elems, ast_.ListOperator.AND),
                ast_.UnaryOperator.NOT,
            ):
                return ast_.ListOp(
                    [ast_.UnaryOp(elem, ast_.UnaryOperator.NOT) for elem in elems],
                    ast_.ListOperator.OR,
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
                    ast_.ListOp([elem], ast_.ListOperator.OR),
                    expression,
                    op_type,
                )
                new_quantifier._bound_identifiers = ast._bound_identifiers
                return new_quantifier

        return None


class GeneratorSelectionCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            # self.select_generator_predicates_or,
            # self.select_generator_predicates_and,
            # self.set_generation,
            # self.generator_selection_and_dummy_reassignment,
            # self.reduce_duplicate_generators,
            self.generator_selection,
            self.equality_separation_and_substitution,
        ]

    def generator_selection(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                expression,
                op_type,
            ):
                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in quantifier (cannot choose generators)")
                    return None

                if all(map(lambda x: isinstance(x, GeneratorSelection), elems)):
                    logger.debug(f"FAILED: all elements in OR quantifier are already GeneratorSelections, no need to select generators")
                    return None

                predicates: list[ast_.ASTNode] = []
                for elem in elems:
                    if (
                        isinstance(elem, ast_.BinaryOp)
                        and elem.op_type == ast_.BinaryOperator.IN
                        and isinstance(elem.right.get_type, ast_.SetType)
                        and isinstance(elem.left, ast_.Identifier)
                        and (not ast.bound or elem.left in ast.bound or ast._env.get(elem.left.name) is None)
                    ):
                        predicates.append(
                            GeneratorSelection(
                                elem,
                                ast_.And([]),
                            )
                        )
                        continue

                    if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
                        logger.debug(
                            f"FAILED: element {elem} is not a ListOp with AND operator (and not a generator itself). Expected predicates to be in disjunctive normal form (an OR of ANDs)"
                        )
                        return None

                    candidate_generators = []
                    other_predicates = []
                    for item in elem.items:
                        if (
                            isinstance(item, ast_.BinaryOp)
                            and item.op_type == ast_.BinaryOperator.IN
                            and isinstance(item.right.get_type, ast_.SetType)
                            and isinstance(item.left, ast_.Identifier)
                            and (not ast.bound or item.left in ast.bound or ast._env.get(item.left.name) is None)
                        ):
                            candidate_generators.append(item)
                        else:
                            other_predicates.append(item)

                    if not candidate_generators:
                        logger.debug(
                            f"FAILED: no candidate generators found in AND predicate (bound variables are {ast.bound}, hidden bound variables are {ast._hidden_bound_identifiers}, free are {ast.free})"
                        )
                        return None

                    selected_generator = candidate_generators[0]
                    other_predicates += candidate_generators[1:]

                    predicates.append(
                        GeneratorSelection(
                            selected_generator,
                            ast_.And.flatten_and_join(
                                other_predicates,
                                ast_.ListOperator.AND,
                            ),
                        )
                    )

                return ast_.Quantifier(
                    ast_.Or(predicates),
                    expression,
                    op_type,
                )
        return None

    def equality_separation_and_substitution(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                expression,
                op_type,
            ) if all(map(lambda x: isinstance(x, GeneratorSelection), elems)):
                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in quantifier (cannot perform substitutions)")
                    return None

                substitution_found = False
                predicates: list[ast_.ASTNode] = []
                for elem in elems:
                    assert isinstance(elem, GeneratorSelection)
                    logger.debug(f"Checking for substitutions in clause with generator {elem.generator}")

                    substitution = None
                    for and_clause in elem.predicates.items:
                        if not isinstance(and_clause, ast_.BinaryOp) or and_clause.op_type != ast_.BinaryOperator.EQUAL:
                            continue
                        logger.debug(f"Checking for substitution in {and_clause}, {and_clause.free}, {and_clause.bound}")
                        if isinstance(and_clause.left, ast_.Identifier) and and_clause.left != elem.generator.left and ast._env.get(and_clause.left.name) is None:
                            substitution = and_clause
                            substitution_found = True
                            break
                        if isinstance(and_clause.right, ast_.Identifier) and and_clause.right != elem.generator.left and ast._env.get(and_clause.right.name) is None:
                            substitution = ast_.Equal(
                                and_clause.right,
                                and_clause.left,
                            )
                            substitution_found = True
                            break

                    if substitution is None:
                        predicates.append(elem)
                        continue

                    predicates.append(
                        elem.find_and_replace(
                            substitution.left,
                            substitution.right,
                        )
                    )

                if not substitution_found:
                    logger.debug(f"FAILED: no substitutions found in OR quantifier (current environment is {ast._env}), free variables are {ast.free}")
                    return None

                return ast_.Quantifier(
                    ast_.Or(predicates),
                    expression,
                    op_type,
                )
        return None

    # def generator_selection_and_dummy_reassignment(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, ast_.ListOperator.OR),
    #             expression,
    #             op_type,
    #         ):
    #             if all(map(lambda x: isinstance(x, GeneratorSelectionAST), elems)):
    #                 logger.debug(f"FAILED: all elements in OR quantifier are already GeneratorSelectionASTs, no need to select generators")
    #                 return None

    #             predicates: list[ast_.ASTNode] = []
    #             for elem in elems:
    #                 if isinstance(elem, ast_.BinaryOp) and elem.op_type == ast_.BinaryOperator.IN:
    #                     predicates.append(
    #                         GeneratorSelectionAST(
    #                             generator=elem,
    #                             assignments=[],
    #                             condition=ast_.And([]),
    #                         )
    #                     )
    #                     continue

    #                 if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
    #                     logger.debug(
    #                         f"FAILED: element {elem} is not a ListOp with AND operator (and not a generator itself). Expected predicates to be in disjunctive normal form (an OR of ANDs)"
    #                     )
    #                     return None

    #                 candidate_generators = []
    #                 equality_assignments = []
    #                 other_predicates = []
    #                 for item in elem.items:
    #                     match item:
    #                         case ast_.BinaryOp(
    #                             ast_.Identifier(_) as x,
    #                             set_type,
    #                             ast_.BinaryOperator.IN,
    #                         ) if isinstance(
    #                             set_type.get_type, ast_.SetType
    #                         ) and (ast.bound is None or x in ast.bound):
    #                             candidate_generators.append(item)
    #                         case ast_.BinaryOp(
    #                             ast_.Identifier(_) as left,
    #                             right,
    #                             eq_type,
    #                         ) if all(
    #                             [
    #                                 eq_type == ast_.BinaryOperator.EQUAL,
    #                                 left in ast.bound,
    #                             ]
    #                         ):
    #                             equality_assignments.append(ast_.Assignment(left, right))
    #                         case ast_.BinaryOp(
    #                             left,
    #                             ast_.Identifier(_) as right,
    #                             eq_type,
    #                         ) if all(
    #                             [
    #                                 eq_type == ast_.BinaryOperator.EQUAL,
    #                                 left in ast.bound,
    #                             ]
    #                         ):
    #                             equality_assignments.append(ast_.Assignment(right, left))
    #                         case _:
    #                             other_predicates.append(item)
    #                 if not candidate_generators:
    #                     logger.debug(f"FAILED: no candidate generators found in AND predicate (bound variables are {ast.bound}, free are {ast.free})")
    #                     return None

    #                 selected_generator = candidate_generators[0]
    #                 other_predicates += candidate_generators[1:]

    #                 predicates.append(
    #                     GeneratorSelectionAST(
    #                         generator=selected_generator,
    #                         assignments=equality_assignments,
    #                         condition=ast_.And.flatten_and_join(other_predicates, ast_.ListOperator.AND),
    #                     )
    #                 )

    #             return
    #                 ast_.Quantifier(
    #                     ast_.Or(predicates),
    #                     expression,
    #                     op_type,
    #                 ),
    #                 ast._env,
    #             )
    #     return None

    # def reduce_duplicate_generators(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, ast_.ListOperator.OR),
    #             expression,
    #             op_type,
    #         ) if all(map(lambda x: isinstance(x, GeneratorSelectionAST), elems)):

    #             reduced_elem = False
    #             condensed_elems: list[ast_.ASTNode] = []
    #             for i in range(len(elems)):
    #                 elem = elems[i]
    #                 assert isinstance(elem, GeneratorSelectionAST)
    #                 for j in range(i + 1, len(elems)):
    #                     other_elem = elems[j]
    #                     assert isinstance(other_elem, GeneratorSelectionAST)

    #                     if elem.generator == other_elem.generator:
    #                         new_elem = GeneratorSelectionAST(
    #                             generator=elem.generator,
    #                             assignments=elem.assignments + other_elem.assignments,
    #                             condition=ast_.And.flatten_and_join([elem.condition, other_elem.condition], ast_.ListOperator.AND),
    #                         )
    #                         condensed_elems.append(new_elem)
    #                         reduced_elem = True
    #                         break
    #                 else:
    #                     condensed_elems.append(elem)

    #             if not reduced_elem:
    #                 logger.debug(f"FAILED: no duplicate generators found in OR quantifier (bound variables are {ast.bound}, free are {ast.free})")
    #                 return None

    #             return
    #                 ast_.Quantifier(
    #                     ast_.ListOp(condensed_elems, ast_.ListOperator.OR),
    #                     expression,
    #                     op_type,
    #                 ),
    #                 ast._env,
    #             )
    #     return None


class PredicateSimplificationDNFCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.simplify_equality,
            self.simplify_and,
            self.reduce_duplicate_generators,
        ]

    def simplify_equality(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(x, y, ast_.BinaryOperator.EQUAL):
                if x == y:
                    return ast_.True_()
        return None

    def simplify_and(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.AND):
                new_elems = []
                for elem in elems:
                    if isinstance(elem, ast_.True_):
                        continue
                    if isinstance(elem, ast_.False_):
                        return ast_.False_()
                    new_elems.append(elem)

                if elems == new_elems:
                    logger.debug("FAILED: no simplification applied to AND list operation (no clauses were removed)")
                    return None

                if not new_elems:
                    return ast_.And([])

                return ast_.ListOp.flatten_and_join(new_elems, ast_.ListOperator.AND)
        return None

    def reduce_duplicate_generators(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR),
                expression,
                op_type,
            ) if all(map(lambda x: isinstance(x, GeneratorSelection), elems)):

                reduced_elem = False
                condensed_elems: list[ast_.ASTNode] = []
                visited_elem_indices = []
                for i in range(len(elems)):
                    if i in visited_elem_indices:
                        continue

                    elem = elems[i]
                    assert isinstance(elem, GeneratorSelection)

                    for j in range(i + 1, len(elems)):
                        other_elem = elems[j]
                        assert isinstance(other_elem, GeneratorSelection)

                        if elem.generator == other_elem.generator:
                            new_elem = GeneratorSelection(
                                generator=elem.generator,
                                predicates=ast_.And(
                                    [
                                        ast_.Or.flatten_and_join(
                                            [elem.predicates, other_elem.predicates],
                                            ast_.ListOperator.OR,
                                        )
                                    ]
                                ),
                            )
                            # elem.copy_and_concat_predicates(other_elem.predicates)
                            condensed_elems.append(new_elem)
                            reduced_elem = True
                            visited_elem_indices.append(j)
                            break
                    else:
                        condensed_elems.append(elem)

                if not reduced_elem:
                    logger.debug(f"FAILED: no duplicate generators found in OR quantifier (bound variables are {ast.bound}, free are {ast.free})")
                    return None

                return ast_.Quantifier(
                    ast_.ListOp(condensed_elems, ast_.ListOperator.OR),
                    expression,
                    op_type,
                )
        return None


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

                return ast_.Statements(new_statements)
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
                if not all(map(lambda x: isinstance(x, GeneratorSelection), predicate.items)):
                    logger.debug(f"FAILED: not all items in predicate are GeneratorSelections (got {predicate.items}). Elements of predicates should be GeneratorSelectionASTs")
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
                    if not isinstance(predicate, GeneratorSelection):
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
                        predicate.copy_and_concat_predicates(past_predicate_inverse),
                        body,
                    )
                    # if_statement._rewrite_generators = [ast._rewrite_generators[i]]
                    # if_statement._bound_by_quantifier_rewrite = ast._bound_by_quantifier_rewrite
                    statements.append(if_statement)
                    past_predicates.append(predicate.flatten())

                return ast_.Statements(
                    statements,
                )

        return None

    def conjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.If(
                GeneratorSelection(generator, predicate),
                body,
            ):
                if not isinstance(generator, ast_.BinaryOp) or generator.op_type != ast_.BinaryOperator.IN or not isinstance(generator.left, ast_.Identifier):
                    logger.debug(f"FAILED: generator is not a valid IN operation (got {generator})")
                    return None

                free_predicates = []
                bound_predicates = []
                for condition_item in predicate.items:
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

                # if assignments:
                #     statement = ast_.Statements(
                #         assignments + [statement],
                #     )

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

                return ast_.Statements([statement])

        return None


SET_REWRITE_COLLECTION: list[type[RewriteCollection]] = [
    SetComprehensionConstructionCollection,
    DisjunctiveNormalFormQuantifierPredicateCollection,
    PredicateSimplificationCollection,
    GeneratorSelectionCollection,
    PredicateSimplificationDNFCollection,
    SetCodeGenerationCollection,
]


class RelationOperatorCollection(RewriteCollection):
    # Todo there should be special rules depending on surjective/injective/totality of relations...
    # Would probably be best to figure them out at this level

    bound_quantifier_variables: set[ast_.Identifier] = field(default_factory=set)

    def apply_all_rules_one_traversal(self, ast):
        bound_quantifier_variables_before = deepcopy(self.bound_quantifier_variables)
        if isinstance(ast, ast_.Quantifier):
            logger.debug(f"Quantifier found with bound variables: {ast.bound}")
            self.bound_quantifier_variables |= ast.bound

        ast = super().apply_all_rules_one_traversal(ast)

        self.bound_quantifier_variables = bound_quantifier_variables_before
        return ast

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.image,
            self.product,
            self.inverse,
            # self.composition,
            self.override,
            self.domain_restriction,
            self.range_restriction,
            self.domain_subtraction,
            self.range_subtraction,
            self.domain,
            self.range,
        ]

    def image(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Image(left, right):
                if not isinstance(left.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of image is not a set type: {left.get_type}")
                    return None
                if not ast_.SetType.is_relation(left.get_type):
                    logger.debug(f"FAILED: left side of image is not a relation type: {left.get_type}")
                    return None
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: right side of image is not a set type: {right.get_type}")
                    return None

                maplet_left = ast_.Identifier(self._get_fresh_identifier_name())
                maplet = ast_.Maplet(
                    maplet_left,
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                set_comprehension = ast_.SetComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, left),
                            ast_.In(maplet.left, right),
                        ]
                    ),
                    maplet.right,
                )
                set_comprehension._bound_identifiers = {maplet_left}

                return set_comprehension

        return None

    def product(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.BinaryOp(
                    maplet_left,
                    maplet_right,
                    ast_.BinaryOperator.MAPLET,
                ),
                ast_.BinaryOp(
                    left,
                    right,
                    ast_.BinaryOperator.CARTESIAN_PRODUCT,
                ),
                ast_.BinaryOperator.IN,
            ):
                if not isinstance(left.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of product is not a set type: {left.get_type}")
                    return None
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: right side of product is not a set type: {right.get_type}")
                    return None

                # Inside quantifier predicate check, similar to membership collapse
                if maplet_left not in self.bound_quantifier_variables:
                    logger.debug(f"FAILED: {maplet_left} appears as a generator variable but is not bound by a quantifier")
                    return None
                if maplet_right not in self.bound_quantifier_variables:
                    logger.debug(f"FAILED: {maplet_right} appears as a generator variable but is not bound by a quantifier")
                    return None

                return ast_.And(
                    [
                        ast_.In(maplet_left, left),
                        ast_.In(maplet_right, right),
                    ]
                )
        return None

    def inverse(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.BinaryOp(
                    maplet_left,
                    maplet_right,
                    ast_.BinaryOperator.MAPLET,
                ),
                ast_.UnaryOp(
                    inner,
                    ast_.UnaryOperator.INVERSE,
                ),
                ast_.BinaryOperator.IN,
            ):
                if not isinstance(inner.get_type, ast_.SetType):
                    logger.debug(f"FAILED: inner side of inverse is not a set/relation type: {inner.get_type}")
                    return None
                if not ast_.SetType.is_relation(inner.get_type):
                    logger.debug(f"FAILED: inner side of inverse is not a relation type: {inner.get_type}")
                    return None

                if maplet_left not in self.bound_quantifier_variables:
                    logger.debug(f"FAILED: {maplet_left} appears as a generator variable but is not bound by a quantifier")
                    return None
                if maplet_right not in self.bound_quantifier_variables:
                    logger.debug(f"FAILED: {maplet_right} appears as a generator variable but is not bound by a quantifier")
                    return None

                return ast_.In(
                    ast_.Maplet(
                        maplet_right,
                        maplet_left,
                    ),
                    inner,
                )
        return None

    def override(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                left,
                right,
                ast_.BinaryOperator.RELATION_OVERRIDING,
            ):
                return ast_.Union(
                    left,
                    ast_.DomainSubtraction(
                        ast_.Call(ast_.Identifier("dom"), [left]),
                        right,
                    ),
                )
        return None

    def domain(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("dom"), [relation]) if isinstance(relation.get_type, ast_.SetType):
                if not ast_.SetType.is_relation(relation.get_type):
                    logger.debug(f"FAILED: {relation} is not a relation type")
                    return None

                maplet = ast_.Maplet(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                set_comprehension = ast_.SetComprehension(
                    ast_.And(
                        [ast_.In(maplet, relation)],
                    ),
                    maplet.left,
                )

                return set_comprehension

        return None

    def range(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("ran"), [relation]) if isinstance(relation.get_type, ast_.SetType):
                if not ast_.SetType.is_relation(relation.get_type):
                    logger.debug(f"FAILED: {relation} is not a relation type")
                    return None

                maplet = ast_.Maplet(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                set_comprehension = ast_.SetComprehension(
                    ast_.And(
                        [ast_.In(maplet, relation)],
                    ),
                    maplet.right,
                )

                return set_comprehension
        return None

    def domain_restriction(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                left,
                right,
                ast_.BinaryOperator.DOMAIN_RESTRICTION,
            ):
                if not isinstance(left.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of domain restriction is not a set type: {left.get_type}")
                    return None
                if not isinstance(right.get_type, ast_.SetType) or not ast_.SetType.is_relation(right.get_type):
                    logger.debug(f"FAILED: right side of domain restriction is not a relation type: {right.get_type}")
                    return None

                maplet = ast_.Maplet(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                return ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, right),
                            ast_.In(maplet.left, left),
                        ],
                    ),
                    maplet,
                )
        return None

    def domain_subtraction(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                left,
                right,
                ast_.BinaryOperator.DOMAIN_SUBTRACTION,
            ):
                if not isinstance(left.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of domain subtraction is not a set type: {left.get_type}")
                    return None

                if not isinstance(right.get_type, ast_.SetType) or not ast_.SetType.is_relation(right.get_type):
                    logger.debug(f"FAILED: right side of domain subtraction is not a relation type: {right.get_type}")
                    return None

                maplet = ast_.Maplet(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                return ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, right),
                            ast_.NotIn(maplet.left, left),
                        ],
                    ),
                    maplet,
                )
        return None

    def range_restriction(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                left,
                right,
                ast_.BinaryOperator.RANGE_RESTRICTION,
            ):
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of range restriction is not a set type: {right.get_type}")
                    return None
                if not isinstance(left.get_type, ast_.SetType) or not ast_.SetType.is_relation(left.get_type):
                    logger.debug(f"FAILED: right side of range restriction is not a relation type: {left.get_type}")
                    return None

                maplet = ast_.Maplet(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                return ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, left),
                            ast_.In(maplet.left, right),
                        ],
                    ),
                    maplet,
                )
        return None

    def range_subtraction(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                left,
                right,
                ast_.BinaryOperator.RANGE_SUBTRACTION,
            ):
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of range subtraction is not a set type: {right.get_type}")
                    return None
                if not isinstance(left.get_type, ast_.SetType) or not ast_.SetType.is_relation(left.get_type):
                    logger.debug(f"FAILED: right side of range subtraction is not a relation type: {left.get_type}")
                    return None

                maplet = ast_.Maplet(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                return ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, left),
                            ast_.NotIn(maplet.left, right),
                        ],
                    ),
                    maplet,
                )
        return None


# Other rewrite rules:
# {x | x \subset S} ~> {x | x in POWERSET(S)}
# ... and other subset operations translated to powerset elements similarly
# POWERSET(S) ~> ?
# {| a, a, a, b, b, c|}
