from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection

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
    current_bound_identifiers: list[ast_.IdentList] = field(default_factory=list)

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
            self.singleton_membership,
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
                new_ast._bound_identifiers = ast_.IdentList([ast_.Identifier(fresh_name)])
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
                new_ast._bound_identifiers = ast_.IdentList([ast_.Identifier(fresh_name)])
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
                new_ast._bound_identifiers = ast_.IdentList([ast_.Identifier(fresh_name)])
                return analysis.add_environments_to_ast(new_ast, ast._env)

        return None

    def singleton_membership(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(ast_.Identifier(_) as x, ast_.Enumeration([elem], _), ast_.BinaryOperator.IN):
                new_ast = ast_.Equal(x, elem)
                return analysis.add_environments_to_ast(new_ast, ast._env)

        return None

    # def membership_collapse(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         # This rewrite rule is really just: f(x) | x in {g(y) | y in S} -> f(g(y)) | y in S,
    #         # but we need to consider the following cases:
    #         # - Comprehensions, Quantifiers, maybe BoolQuantifiers, maybe lambda defs?
    #         # - generators mixed with predicates (combined with ANDS. ex: f(x) | x in {g(y) | y in S} AND p(x) -> f(g(y)) | y in S AND p(g(y)) )
    #         #
    #         # No occurrences of x should be in the resulting ASTNode (TODO check this)
    #         # case ast_.Quantifier(
    #         #     ast_.BinaryOp(
    #         #         ast_.Identifier(_) as x,
    #         #         ast_.Quantifier(
    #         #             ast_.BinaryOp(
    #         #                 ast_.Identifier(_) as y,
    #         #                 inner_set_type,
    #         #                 ast_.BinaryOperator.IN,
    #         #             ),
    #         #             inner_expression,
    #         #             inner_op_type,
    #         #         ) as outer_set_type,
    #         #         ast_.BinaryOperator.IN,
    #         #     ),
    #         #     outer_expression,
    #         #     outer_op_type,
    #         # ):
    #         #     if any(
    #         #         [
    #         #             outer_op_type != inner_op_type,
    #         #             not isinstance(outer_set_type, ast_.SetType),
    #         #             # Be conservative here, only try rewriting on one free variable
    #         #             len(ast.free) != 1,
    #         #             x != next(iter(ast.free)),  # Generator must match free variable
    #         #         ]
    #         #     ):
    #         #         return None

    #         #     # change expression from f(x) to f(g(y))
    #         #     outer_expression = outer_expression.find_and_replace(x, inner_expression)

    #         #     return ast_.Quantifier(
    #         #         ast_.In(y, inner_set_type),
    #         #         outer_expression,
    #         #         outer_op_type,  # type: ignore
    #         #     )

    #         case ast_.Quantifier(
    #             ast_.ListOp(_, ast_.ListOperator.AND) as outer_predicate,
    #             outer_expression,
    #             outer_op_type,
    #         ):

    #             if len(ast.free) != 1:
    #                 return None

    #             candidate_generators, predicates = outer_predicate.separate_candidate_generators_from_predicates(ast.free)

    #             collapsing_generators: list[tuple[ast_.Identifier, ast_.Identifier, ast_.ASTNode, ast_.ASTNode]] = []
    #             for candidate_generator in candidate_generators:
    #                 match candidate_generator:
    #                     case ast_.BinaryOp(
    #                         ast_.Identifier(_) as x,
    #                         ast_.Quantifier(
    #                             ast_.And(
    #                                 [
    #                                     ast_.BinaryOp(
    #                                         ast_.Identifier(_) as y,
    #                                         inner_set_type,
    #                                         ast_.BinaryOperator.IN,
    #                                     )
    #                                 ]
    #                             ),
    #                             inner_expression,
    #                             inner_op_type,
    #                         ) as outer_set_type,
    #                         ast_.BinaryOperator.IN,
    #                     ) if all(
    #                         [
    #                             outer_op_type == inner_op_type,
    #                             isinstance(outer_set_type, ast_.SetType),
    #                             # Be conservative here, only try rewriting on one free variable
    #                             x == next(iter(ast.free)),  # Generator must match free variable
    #                         ]
    #                     ):
    #                         collapsing_generators.append(
    #                             (
    #                                 x,
    #                                 y,
    #                                 inner_expression,
    #                                 inner_set_type,
    #                             )
    #                         )
    #                     case _:
    #                         predicates.append(candidate_generator)

    #             # Be conservative with rewrites, only one generator can be collapsed
    #             if len(collapsing_generators) != 1:
    #                 return None
    #             x, y, inner_expression, inner_set_type = collapsing_generators[0]

    #             new_predicates = []
    #             for predicate in predicates:
    #                 # Change predicate from p(x) to p(g(y))
    #                 new_predicate = predicate.find_and_replace(x, inner_expression)
    #                 new_predicates.append(new_predicate)

    #             outer_expression = outer_expression.find_and_replace(x, inner_expression)

    #             return ast_.Quantifier(
    #                 ast_.And([ast_.In(y, inner_set_type)] + new_predicates),
    #                 outer_expression,
    #                 outer_op_type,  # type: ignore # TODO check for type error here
    #             )

    #         # case ast_.Quantifier(
    #         #     ast_.ListOp(_, ast_.ListOperator.OR) as outer_predicate,
    #         #     outer_expression,
    #         #     outer_op_type,
    #         # ):
    #         #     if len(ast.free) != 1:  # should these be bound?
    #         #         return None

    #         #     new_or_clauses = []
    #         #     for or_clause in outer_predicate.items:
    #         #         match or_clause:
    #         #             case ast_.In(
    #         #                 a,
    #         #                 ast_.Quantifier(
    #         #                     inner_predicate,
    #         #                     inner_expression,
    #         #                     inner_op_type,
    #         #                 ),
    #         #             ):
    #         #                 if a not in ast.free:
    #         #                     return None

    #         #             case ast_.ListOp(_, ast_.ListOperator.AND) as inner_predicate:
    #         #                 pass

    #     return None

    # def membership_collapse_2(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     match ast:
    #         case ast_.Quantifier(
    #             ast_.ListOp(elems, list_op_type) as outer_predicate,
    #             outer_expression,
    #             outer_op_type,
    #         ) if outer_op_type.is_collection_operator():
    #             # Idea, look inside the elements, lifting nested comprehensions

    #             def match_and_lift(elem: ast_.ASTNode) -> ast_.ASTNode | None:
    #                 match elem:
    #                     # Elem is of the form x in {y | y in S and p(y)}
    #                     case ast_.In(
    #                         ast_.Identifier(_) as x,
    #                         ast_.Quantifier(
    #                             inner_predicate,
    #                             inner_expression,
    #                             inner_op_type,
    #                         ),
    #                     ) if inner_op_type.is_collection_operator():
    #                         if x not in ast.free:
    #                             return None
    #                         # New form is y in S and p(x)
    #                         return ast_.ListOp(
    #                             [
    #                                 inner_predicate,
    #                                 ast_.Equal(x, inner_expression),
    #                             ],
    #                             ast_.ListOperator.AND,
    #                         )
    #                 return None

    #             return ast_.Quantifier(
    #                 ast_.ListOp.flatten_and_join(
    #                     [outer_predicate.find_and_replace_with_func(match_and_lift)],
    #                     list_op_type,
    #                 ),
    #                 outer_expression,
    #                 outer_op_type,
    #             )

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
                    self.current_bound_identifiers[-1].items.extend(inner_quantifier._bound_identifiers.items)

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
                non_or_elems = non_or_elems + or_elems[1:]
                for item in or_elem_to_distribute.items:
                    new_elems.append(
                        ast_.And(
                            [item] + non_or_elems,
                        )
                    )

                return analysis.add_environments_to_ast(ast_.Or(new_elems), ast._env)

        return None


# class SetOptimizationCollection(RewriteCollection):


#     def bubble_up_generators(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
#         match ast:
#             case
#         return None
class GeneratorSelectionCollection(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.select_generator_predicates_or,
            self.select_generator_predicates_and,
            # self.set_generation,
        ]

    def select_generator_predicates_or(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(ast_.ListOp(elems, ast_.ListOperator.OR), _, _):
                if ast._selected_generators is not None:
                    return None

                generators = []
                for elem in elems:
                    if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
                        continue

                    candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.bound)
                    if not candidate_generators:
                        logger.debug(f"FAILED: no candidate generators found in OR predicate (bound variables are {ast.bound}, free are {ast.free})")
                        return None

                    # TODO For now, just manually choose the first available generator
                    generators.append(candidate_generators[0])

                ast._selected_generators = generators
                return ast
        return None

    def select_generator_predicates_and(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(ast_.ListOp(_, ast_.ListOperator.AND) as elem, _, _):
                if ast._selected_generators is not None:
                    return None

                candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.bound)
                if not candidate_generators:
                    logger.debug(f"FAILED: no candidate generators found in AND predicate (bound variables are {ast.bound}, free are {ast.free})")
                    return None

                ast._selected_generators = [candidate_generators[0]]
                return ast
        return None


class SetCodeGenerationCollection(RewriteCollection):

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.summation_2,
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

    def summation_2(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                predicate,
                expression,
                ast_.QuantifierOperator.SUM,
            ):
                # Idea - put everything in an if statement to start
                counter = ast_.Identifier(self._get_fresh_identifier_name())

                assert ast._env is not None, f"Environment should not be None (in summation, ast {ast})"
                ast._env.put(counter.name, ast_.BaseSimileType.Int)

                if_statement = ast_.If(
                    predicate,
                    ast_.Assignment(
                        counter,
                        ast_.Add(
                            counter,
                            expression,
                        ),
                    ),
                )
                if_statement._rewrite_generators = ast._selected_generators
                if_statement._bound_by_quantifier_rewrite = ast.bound

                return analysis.add_environments_to_ast(
                    ast_.Statements(
                        [
                            ast_.Assignment(
                                counter,
                                ast_.Int("0"),
                            ),
                            if_statement,
                        ]
                    ),
                    ast._env,
                )
                # new_statement._env = ast._env
                # assert new_statement._env is not None, f"Environment should not be None (in summation, ast {new_statement})"
                # new_statement._env.put(counter.name, ast_.BaseSimileType.Int)
                # return new_statement

        return None

    def disjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.If(
                ast_.ListOp(predicates, ast_.ListOperator.OR),
                body,
            ):
                if ast._rewrite_generators is None:
                    logger.debug(f"FAILED: no candidate generators found in disjunctive predicate")
                    return None
                if ast._bound_by_quantifier_rewrite is None:
                    logger.debug(f"FAILED: no bound variables found in disjunctive predicate")
                    return None

                statements: list[ast_.ASTNode] = []
                past_predicates: list[ast_.ASTNode] = []

                assert len(ast._rewrite_generators) == len(
                    predicates
                ), f"Every OR predicate should have a generator, but got {len(ast._rewrite_generators)} generators and {len(predicates)} predicates"

                # Every predicate should have a generator matching 1-1 from the GeneratorSelectionCollection
                for i, predicate in enumerate(predicates):
                    past_predicate_inverse = []
                    if past_predicates:
                        past_predicate_inverse = [
                            ast_.Not(
                                ast_.ListOp.flatten_and_join(
                                    past_predicates,
                                    ast_.ListOperator.AND,
                                ),
                            )
                        ]

                    if_statement = ast_.If(
                        ast_.ListOp.flatten_and_join(
                            [predicate] + past_predicate_inverse,
                            ast_.ListOperator.AND,
                        ),
                        body,
                    )
                    if_statement._rewrite_generators = [ast._rewrite_generators[i]]
                    if_statement._bound_by_quantifier_rewrite = ast._bound_by_quantifier_rewrite
                    statements.append(if_statement)
                    past_predicates.append(predicate)
                return analysis.add_environments_to_ast(
                    ast_.Statements(
                        statements,
                    ),
                    ast._env,
                )
        return None

    def conjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.If(
                ast_.ListOp(predicates_, ast_.ListOperator.AND) as predicate,
                body,
            ):
                if ast._rewrite_generators is None:
                    logger.debug(f"FAILED: no candidate generators found in conjunctive predicate")
                    return None
                if ast._bound_by_quantifier_rewrite is None:
                    logger.debug(f"FAILED: no bound variables found in conjunctive predicate")
                    return None

                if not ast._rewrite_generators:
                    logger.debug(f"FAILED: no candidate generators found in conjunctive predicate")
                    return None
                if len(ast._rewrite_generators) != 1:
                    logger.debug(f"FAILED: more than one candidate generator found in conjunctive predicate: {ast._rewrite_generators}")
                    return None

                generator = ast._rewrite_generators[0]
                if not isinstance(generator, ast_.BinaryOp) or generator.op_type != ast_.BinaryOperator.IN or not isinstance(generator.left, ast_.Identifier):
                    logger.debug(f"FAILED: generator is not a valid IN operation (got {generator})")
                    return None

                predicates = list(filter(lambda x: x != generator, predicates_))
                free_predicates = list(filter(lambda x: not x.contains_item(generator.left), predicates))
                bound_predicates = list(filter(lambda x: x.contains_item(generator.left), predicates))

                statement: ast_.ASTNode = body
                if bound_predicates:
                    # Handle bound-to-bound variable equality checks as assignments
                    if_statement_bound_predicates = []

                    assignment_statements: list[ast_.ASTNode] = []
                    for bound_predicate in bound_predicates:
                        if not isinstance(bound_predicate, ast_.BinaryOp) or not bound_predicate.op_type == ast_.BinaryOperator.EQUAL:
                            if_statement_bound_predicates.append(bound_predicate)
                            continue

                        for bound_var in ast._bound_by_quantifier_rewrite:
                            if bound_var not in bound_predicate.left.free and generator.left not in bound_predicate.right.free:
                                continue

                            assignment_statements.append(
                                ast_.Assignment(
                                    bound_predicate.left,
                                    bound_predicate.right,  # The right side should have the generator variable
                                ),
                            )
                            break

                    if if_statement_bound_predicates:
                        statement = ast_.Statements(
                            assignment_statements
                            + [
                                ast_.If(
                                    ast_.ListOp.flatten_and_join(
                                        if_statement_bound_predicates,
                                        ast_.ListOperator.AND,
                                    ),
                                    statement,
                                )
                            ]
                        )
                    else:
                        statement = ast_.Statements(
                            assignment_statements + [statement],
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
