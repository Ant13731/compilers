from __future__ import annotations
from typing import Callable
from dataclasses import dataclass

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection

# NOTE: REWRITE RULES MUST ALWAYS USE THE PARENT FORM FOR STRUCTURAL MATCHING (ex. BinaryOp instead of Add)


@dataclass
class SetRewriteCollection(RewriteCollection):
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

    def membership_collapse(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # This rewrite rule is really just: f(x) | x in {g(y) | y in S} -> f(g(y)) | y in S,
            # but we need to consider the following cases:
            # - Comprehensions, Quantifiers, maybe BoolQuantifiers, maybe lambda defs?
            # - generators mixed with predicates (combined with ANDS. ex: f(x) | x in {g(y) | y in S} AND p(x) -> f(g(y)) | y in S AND p(g(y)) )
            #
            # No occurrences of x should be in the resulting ASTNode (TODO check this)
            # case ast_.Quantifier(
            #     ast_.BinaryOp(
            #         ast_.Identifier(_) as x,
            #         ast_.Quantifier(
            #             ast_.BinaryOp(
            #                 ast_.Identifier(_) as y,
            #                 inner_set_type,
            #                 ast_.BinaryOperator.IN,
            #             ),
            #             inner_expression,
            #             inner_op_type,
            #         ) as outer_set_type,
            #         ast_.BinaryOperator.IN,
            #     ),
            #     outer_expression,
            #     outer_op_type,
            # ):
            #     if any(
            #         [
            #             outer_op_type != inner_op_type,
            #             not isinstance(outer_set_type, ast_.SetType),
            #             # Be conservative here, only try rewriting on one free variable
            #             len(ast.free) != 1,
            #             x != next(iter(ast.free)),  # Generator must match free variable
            #         ]
            #     ):
            #         return None

            #     # change expression from f(x) to f(g(y))
            #     outer_expression = outer_expression.find_and_replace(x, inner_expression)

            #     return ast_.Quantifier(
            #         ast_.In(y, inner_set_type),
            #         outer_expression,
            #         outer_op_type,  # type: ignore
            #     )

            case ast_.Quantifier(
                ast_.ListOp(_, ast_.ListOperator.AND) as outer_predicate,
                outer_expression,
                outer_op_type,
            ):

                if len(ast.free) != 1:
                    return None

                candidate_generators, predicates = outer_predicate.separate_candidate_generators_from_predicates(ast.free)

                collapsing_generators: list[tuple[ast_.Identifier, ast_.Identifier, ast_.ASTNode, ast_.ASTNode]] = []
                for candidate_generator in candidate_generators:
                    match candidate_generator:
                        case ast_.BinaryOp(
                            ast_.Identifier(_) as x,
                            ast_.Quantifier(
                                ast_.And(
                                    [
                                        ast_.BinaryOp(
                                            ast_.Identifier(_) as y,
                                            inner_set_type,
                                            ast_.BinaryOperator.IN,
                                        )
                                    ]
                                ),
                                inner_expression,
                                inner_op_type,
                            ) as outer_set_type,
                            ast_.BinaryOperator.IN,
                        ) if all(
                            [
                                outer_op_type == inner_op_type,
                                isinstance(outer_set_type, ast_.SetType),
                                # Be conservative here, only try rewriting on one free variable
                                x == next(iter(ast.free)),  # Generator must match free variable
                            ]
                        ):
                            collapsing_generators.append(
                                (
                                    x,
                                    y,
                                    inner_expression,
                                    inner_set_type,
                                )
                            )
                        case _:
                            predicates.append(candidate_generator)

                # Be conservative with rewrites, only one generator can be collapsed
                if len(collapsing_generators) != 1:
                    return None
                x, y, inner_expression, inner_set_type = collapsing_generators[0]

                new_predicates = []
                for predicate in predicates:
                    # Change predicate from p(x) to p(g(y))
                    new_predicate = predicate.find_and_replace(x, inner_expression)
                    new_predicates.append(new_predicate)

                outer_expression = outer_expression.find_and_replace(x, inner_expression)

                return ast_.Quantifier(
                    ast_.And([ast_.In(y, inner_set_type)] + new_predicates),
                    outer_expression,
                    outer_op_type,  # type: ignore # TODO check for type error here
                )

            # case ast_.Quantifier(
            #     ast_.ListOp(_, ast_.ListOperator.OR) as outer_predicate,
            #     outer_expression,
            #     outer_op_type,
            # ):
            #     if len(ast.free) != 1:  # should these be bound?
            #         return None

            #     new_or_clauses = []
            #     for or_clause in outer_predicate.items:
            #         match or_clause:
            #             case ast_.In(
            #                 a,
            #                 ast_.Quantifier(
            #                     inner_predicate,
            #                     inner_expression,
            #                     inner_op_type,
            #                 ),
            #             ):
            #                 if a not in ast.free:
            #                     return None

            #             case ast_.ListOp(_, ast_.ListOperator.AND) as inner_predicate:
            #                 pass

        return None

    def membership_collapse_2(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, list_op_type) as outer_predicate,
                outer_expression,
                outer_op_type,
            ) if outer_op_type.is_collection_operator():
                # Idea, look inside the elements, lifting nested comprehensions

                def match_and_lift(elem: ast_.ASTNode) -> ast_.ASTNode | None:
                    match elem:
                        # Elem is of the form x in {y | y in S and p(y)}
                        case ast_.In(
                            ast_.Identifier(_) as x,
                            ast_.Quantifier(
                                inner_predicate,
                                inner_expression,
                                inner_op_type,
                            ),
                        ) if inner_op_type.is_collection_operator():
                            if x not in ast.free:
                                return None
                            # New form is y in S and p(x)
                            return ast_.ListOp(
                                [
                                    inner_predicate,
                                    ast_.Equal(x, inner_expression),
                                ],
                                ast_.ListOperator.AND,
                            )
                    return None

                return ast_.Quantifier(
                    ast_.ListOp.flatten_and_join(
                        [outer_predicate.find_and_replace_with_func(match_and_lift)],
                        list_op_type,
                    ),
                    outer_expression,
                    outer_op_type,
                )

        return None

    def set_generation(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.OR) as predicate,
                expression,
                op_type,
            ):
                ret_id = ast_.Identifier(self._get_fresh_identifier_name())

                statements: list[ast_.ASTNode] = [
                    ast_.Assignment(
                        ret_id,
                        ast_.SetEnumeration(
                            [],
                            op_type=ast_.CollectionOperator.SET,
                        ),
                    ),
                ]

                for elem in elems:
                    if isinstance(elem, ast_.ListOp) and elem.op_type == ast_.ListOperator.AND:
                        candidate_generators, predicates = elem.separate_candidate_generators_from_predicates(ast.free)
                    elif isinstance(elem, ast_.BinaryOp) and elem.op_type == ast_.BinaryOperator.IN:
                        candidate_generators = [elem]
                        predicates = []
                    else:
                        return None

                    if not candidate_generators:
                        return None

                    candidate_generator = next(iter(candidate_generators))
                    if candidate_generator.op_type != ast_.BinaryOperator.IN:
                        return None
                    if not isinstance(candidate_generator.left, ast_.Identifier):
                        return None

                    statements.append(
                        ast_.For(
                            ast_.IdentList([candidate_generator.left]),
                            candidate_generator.right,
                            ast_.Statements(
                                [
                                    ast_.If(
                                        ast_.And(predicates + candidate_generators[1:]),
                                        ast_.Call(
                                            ast_.StructAccess(
                                                ret_id,
                                                ast_.Identifier("add"),
                                            ),
                                            [expression],
                                        ),
                                        ast_.None_(),
                                    )
                                ]
                            ),
                        )
                    )
            case ast_.Quantifier(
                ast_.ListOp(elems, ast_.ListOperator.AND) as predicate,
                expression,
                op_type,
            ):
                ret_id = ast_.Identifier(self._get_fresh_identifier_name())

                statements = [
                    ast_.Assignment(
                        ret_id,
                        ast_.SetEnumeration(
                            [],
                            op_type=ast_.CollectionOperator.SET,
                        ),
                    ),
                ]

                candidate_generators, predicates = predicate.separate_candidate_generators_from_predicates(ast.free)

                if not candidate_generators:
                    return None

                candidate_generator = next(iter(candidate_generators))
                if candidate_generator.op_type != ast_.BinaryOperator.IN:
                    return None
                if not isinstance(candidate_generator.left, ast_.Identifier):
                    return None

                statements.append(
                    ast_.For(
                        ast_.IdentList([candidate_generator.left]),
                        candidate_generator.right,
                        ast_.Statements(
                            [
                                ast_.If(
                                    ast_.And(predicates + candidate_generators[1:]),
                                    ast_.Call(
                                        ast_.StructAccess(
                                            ret_id,
                                            ast_.Identifier("add"),
                                        ),
                                        [expression],
                                    ),
                                    ast_.None_(),
                                )
                            ]
                        ),
                    )
                )

        return None

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
