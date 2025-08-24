from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from copy import deepcopy

from loguru import logger

from src.mod import ast_
from src.mod import analysis
from src.mod.optimizer.rewrite_collection import RewriteCollection
from src.mod.optimizer.intermediate_ast import (
    GeneratorSelection,
    CombinedGeneratorSelection,
    SingleGeneratorSelection,
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

    def bag_predicate_operations(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # Idea, if we want to add some compilation-time optimizations (like union of two set enums into one set enum),
            # we can just add those rules here
            case ast_.BinaryOp(left, right, op_type) if op_type in (
                ast_.BinaryOperator.UNION,
                ast_.BinaryOperator.INTERSECTION,
                ast_.BinaryOperator.ADD,
                ast_.BinaryOperator.SUBTRACT,
            ):
                if not isinstance(left.get_type, ast_.SetType) or not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: at least one union child is not a set type: {left.get_type}, {right.get_type}")
                    return None

                if not ast_.SetType.is_bag(left.get_type) or not ast_.SetType.is_bag(right.get_type):
                    logger.debug(f"FAILED: Both operands must be bags")
                    return None

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                match op_type:
                    case ast_.BinaryOperator.SUBTRACT:
                        logger.debug("WARNING: Subtracting bags not fully correctly implemented (subtraction doesn't work)")
                        # TODO see if this method is more efficient?
                        new_ast = ast_.BagComprehension(
                            ast_.And(
                                [
                                    ast_.In(
                                        maplet.left,
                                        ast_.Call(ast_.Identifier("dom"), [left]),
                                    ),
                                    ast_.Equal(
                                        maplet.left,
                                        ast_.Subtract(
                                            ast_.Call(
                                                ast_.Identifier("pop"),
                                                [
                                                    ast_.Image(left, maplet.left),
                                                ],
                                            ),
                                            ast_.Call(
                                                ast_.Identifier("pop_default"),
                                                [
                                                    ast_.Image(right, maplet.left),
                                                    ast_.Int("0"),
                                                ],
                                            ),
                                        ),
                                    ),
                                    ast_.GreaterThan(maplet.right, ast_.Int("0")),
                                ]
                            ),
                            maplet,
                        )
                        new_ast._bound_identifiers = {maplet.left}
                        return new_ast
                    case ast_.BinaryOperator.UNION:
                        func_name = "max"
                        generator = ast_.In(
                            maplet.left,
                            ast_.Union(
                                ast_.Call(ast_.Identifier("dom"), [left]),
                                ast_.Call(ast_.Identifier("dom"), [right]),
                            ),
                        )
                        additional_cond: ast_.ASTNode = ast_.True_()
                    case ast_.BinaryOperator.ADD:
                        func_name = "sum"
                        generator = ast_.In(
                            maplet.left,
                            ast_.Union(
                                ast_.Call(ast_.Identifier("dom"), [left]),
                                ast_.Call(ast_.Identifier("dom"), [right]),
                            ),
                        )
                        additional_cond = ast_.True_()
                    case ast_.BinaryOperator.INTERSECTION:
                        func_name = "min"
                        generator = ast_.In(
                            maplet.left,
                            ast_.Intersection(
                                ast_.Call(ast_.Identifier("dom"), [left]),
                                ast_.Call(ast_.Identifier("dom"), [right]),
                            ),
                        )
                        additional_cond = ast_.GreaterThan(maplet.right, ast_.Int("0"))

                new_ast = ast_.BagComprehension(
                    ast_.And(
                        [
                            generator,
                            ast_.Equal(
                                maplet.left,
                                ast_.Call(
                                    ast_.Identifier(func_name),
                                    [
                                        ast_.Union(
                                            ast_.Image(left, maplet.left),
                                            ast_.Image(right, maplet.left),
                                        ),
                                    ],
                                ),
                            ),
                            additional_cond,
                        ]
                    ),
                    maplet,
                )
                new_ast._bound_identifiers = {maplet.left}
                return new_ast
        return None


@dataclass
class BuiltinFunctions(RewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.cardinality,
            self.domain,
            self.range_,
            self.override,
            self.range_restriction,
            self.range_subtraction,
            self.domain_restriction,
            self.domain_subtraction,
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
            case ast_.Call(ast_.Identifier("dom"), [relation]):
                if not isinstance(relation.get_type, ast_.SetType):
                    logger.warning(f"Argument to domain {relation} is not a set")
                    return None

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                new_ast = ast_.SetComprehension(
                    ast_.And([ast_.In(maplet, relation)]),
                    maplet.left,
                )
                new_ast._bound_identifiers = {maplet}
                return new_ast
        return None

    def range_(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("ran"), [relation]):
                if not isinstance(relation.get_type, ast_.SetType):
                    logger.warning(f"Argument to range {relation} is not a set")
                    return None

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                new_ast = ast_.SetComprehension(
                    ast_.And([ast_.In(maplet, relation)]),
                    maplet.right,
                )
                new_ast._bound_identifiers = {maplet}
                return new_ast
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

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                new_ast = ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, right),
                            ast_.In(maplet.left, left),
                        ],
                    ),
                    maplet,
                )

                new_ast._bound_identifiers = {maplet}
                return new_ast
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

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                new_ast = ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, right),
                            ast_.NotIn(maplet.left, left),
                        ],
                    ),
                    maplet,
                )
                new_ast._bound_identifiers = {maplet}
                return new_ast
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

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                new_ast = ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, left),
                            ast_.In(maplet.left, right),
                        ],
                    ),
                    maplet,
                )
                new_ast._bound_identifiers = {maplet}
                return new_ast
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

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )

                new_ast = ast_.RelationComprehension(
                    ast_.And(
                        [
                            ast_.In(maplet, left),
                            ast_.NotIn(maplet.left, right),
                        ],
                    ),
                    maplet,
                )
                new_ast._bound_identifiers = {maplet}
                return new_ast
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

    def sum(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("sum"), [arg]):
                if not isinstance(arg.get_type, ast_.SetType):
                    logger.debug(f"FAILED: argument of sum is not a set type: {arg.get_type}")
                    return None
                if not isinstance(arg.get_type.element_type, ast_.BaseSimileType) or not arg.get_type.element_type.is_numeric():
                    logger.debug(f"FAILED: element type of sum argument is not a numeric type: {arg.get_type.element_type}")
                    return None

                iterator = ast_.Identifier(self._get_fresh_identifier_name())
                new_ast = ast_.Sum(
                    ast_.And([ast_.In(iterator, arg)]),
                    iterator,
                )
                new_ast._bound_identifiers = {iterator}
                return new_ast
        return None

    def bag_size(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(ast_.Identifier("size"), [arg]):
                if not isinstance(arg.get_type, ast_.SetType) or not ast_.SetType.is_bag(arg.get_type):
                    logger.debug(f"FAILED: argument of size is not a bag type: {arg.get_type}")
                    return None

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
                    ast_.Identifier(self._get_fresh_identifier_name()),
                )
                new_ast = ast_.Sum(
                    ast_.And([ast_.In(maplet, arg)]),
                    maplet.right,
                )
                new_ast._bound_identifiers = {maplet}
                return new_ast
        return None


@dataclass
class InsideQuantifierRewriteCollection(RewriteCollection):
    bound_quantifier_variables: set[ast_.Identifier] = field(default_factory=set)
    current_bound_identifiers: list[set[ast_.Identifier | ast_.MapletIdentifier]] = field(default_factory=list)

    def apply_all_rules_one_traversal(self, ast):
        bound_quantifier_variables_before = deepcopy(self.bound_quantifier_variables)
        if isinstance(ast, ast_.Quantifier):
            logger.debug(f"Quantifier found with bound variables: {ast.bound}")
            self.bound_quantifier_variables |= ast.flatten_bound_identifiers()
            self.current_bound_identifiers.append(ast._bound_identifiers)

        ast = super().apply_all_rules_one_traversal(ast)

        self.bound_quantifier_variables = bound_quantifier_variables_before
        if self.current_bound_identifiers and hasattr(ast, "_bound_identifiers"):
            if ast._bound_identifiers != self.current_bound_identifiers:  # type: ignore
                # If the current bound identifiers are set, we need to update the AST's bound identifiers
                ast._bound_identifiers = self.current_bound_identifiers.pop()  # type: ignore
            else:
                self.current_bound_identifiers.pop()
        return ast

    def inside_quantifier(self) -> bool:
        if self.bound_quantifier_variables:
            return True
        return False


@dataclass
class ComprehensionConstructionCollection(InsideQuantifierRewriteCollection):
    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.image,
            self.set_predicate_operations,
            self.product,
            self.inverse,
            self.composition,
            # self.singleton_membership,
            self.membership_collapse,
        ]

    def product(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.MapletIdentifier(_, _) as maplet,
                ast_.BinaryOp(left, right, ast_.BinaryOperator.CARTESIAN_PRODUCT),
                ast_.BinaryOperator.IN,
            ) if self.inside_quantifier():
                # Inside quantifier predicate check, similar to membership collapse
                if not maplet.flatten().issubset(self.bound_quantifier_variables):
                    logger.debug(f"FAILED: {maplet} appears as a generator variable but is not bound by a quantifier")
                    return None

                if not isinstance(left.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of product is not a set type: {left.get_type}")
                    return None
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: right side of product is not a set type: {right.get_type}")
                    return None

                # TODO is this really the correct move? nested quantifiers may cause trouble...
                self.current_bound_identifiers[-1].remove(maplet)
                self.current_bound_identifiers[-1].update({maplet.left, maplet.right})

                return ast_.And(
                    [
                        ast_.In(maplet.left, left),
                        ast_.In(maplet.right, right),
                    ]
                )
        return None

    def inverse(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.MapletIdentifier(_, _) as maplet,
                ast_.UnaryOp(
                    inner,
                    ast_.UnaryOperator.INVERSE,
                ),
                ast_.BinaryOperator.IN,
            ) if self.inside_quantifier():
                if not maplet.flatten().issubset(self.bound_quantifier_variables):
                    logger.debug(f"FAILED: {maplet} appears as a generator variable but is not bound by a quantifier")
                    return None

                if not isinstance(inner.get_type, ast_.SetType):
                    logger.debug(f"FAILED: inner side of inverse is not a set/relation type: {inner.get_type}")
                    return None
                if not ast_.SetType.is_relation(inner.get_type):
                    logger.debug(f"FAILED: inner side of inverse is not a relation type: {inner.get_type}")
                    return None

                rev_maplet = ast_.MapletIdentifier(maplet.right, maplet.left)
                self.current_bound_identifiers[-1].remove(maplet)
                self.current_bound_identifiers[-1].update({rev_maplet})

                return ast_.In(rev_maplet, inner)
        return None

    def composition(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.BinaryOp(
                ast_.MapletIdentifier(_, _) as maplet,
                ast_.BinaryOp(
                    left,
                    right,
                    ast_.BinaryOperator.COMPOSITION,
                ),
                ast_.BinaryOperator.IN,
            ) if self.inside_quantifier():
                if not maplet.flatten().issubset(self.bound_quantifier_variables):
                    logger.debug(f"FAILED: {maplet} appears as a generator variable but is not bound by a quantifier")
                    return None

                if not isinstance(left.get_type, ast_.SetType):
                    logger.debug(f"FAILED: left side of composition is not a set type: {left.get_type}")
                    return None
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: right side of composition is not a set type: {right.get_type}")
                    return None

                fresh_var_left = ast_.Identifier(self._get_fresh_identifier_name())
                fresh_var_right = ast_.Identifier(self._get_fresh_identifier_name())

                maplet_left = ast_.MapletIdentifier(maplet.left, fresh_var_left)
                maplet_right = ast_.MapletIdentifier(fresh_var_right, maplet.right)

                self.current_bound_identifiers[-1].remove(maplet)
                self.current_bound_identifiers[-1].update({maplet_left, maplet_right})

                return ast_.And(
                    [
                        ast_.In(maplet_left, left),
                        ast_.In(maplet_right, right),
                        ast_.Equal(fresh_var_left, fresh_var_right),
                    ]
                )
        return None

    def image(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Image(left, right):
                if not isinstance(left.get_type, ast_.SetType) or not ast_.SetType.is_relation(left.get_type):
                    logger.debug(f"FAILED: left side of image is not a relation type: {left.get_type}")
                    return None
                if not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: right side of image is not a set type: {right.get_type}")
                    return None

                maplet = ast_.MapletIdentifier(
                    ast_.Identifier(self._get_fresh_identifier_name()),
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
                set_comprehension._bound_identifiers = {maplet}

                return set_comprehension

        return None

    def functional_image(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Call(
                rel,
                [arg],
            ) if isinstance(
                rel.get_type, ast_.SetType
            ) and ast_.SetType.is_relation(rel.get_type):
                if rel.get_type.relation_subtype is None:
                    logger.debug(f"FAILED: relation {rel} does not have a functional subtype - functional image may be undefined (disabled for now)")
                    return None
                if not rel.get_type.relation_subtype.total:
                    logger.debug(f"FAILED: relation {rel} is not total - functional image may return nothing (disabled for now)")
                    return None
                if not rel.get_type.relation_subtype.one_to_many:
                    logger.debug(f"FAILED: relation {rel} is not one-to-many - functional image may return multiple values (disabled for now)")
                    return None

                return ast_.Call(ast_.Identifier("pop"), [ast_.Image(rel, arg)])

        return None

    def set_predicate_operations(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # Idea, if we want to add some compilation-time optimizations (like union of two set enums into one set enum),
            # we can just add those rules here
            case ast_.BinaryOp(left, right, op_type) if op_type in (
                ast_.BinaryOperator.UNION,
                ast_.BinaryOperator.INTERSECTION,
                ast_.BinaryOperator.DIFFERENCE,
            ):
                if not isinstance(left.get_type, ast_.SetType) or not isinstance(right.get_type, ast_.SetType):
                    logger.debug(f"FAILED: at least one union child is not a set type: {left.get_type}, {right.get_type}")
                    return None

                if ast_.SetType.is_bag(left.get_type) and ast_.SetType.is_bag(right.get_type):
                    logger.debug(f"FAILED: Union/Intersection/Difference operations must use bag_predicate_operations rule (both operands are bags)")
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
                if self.inside_quantifier() and x not in self.bound_quantifier_variables:
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
            case ast_.Quantifier(ast_.ListOp(_, ast_.ListOperator.OR) as elem, expression, op_type):
                return None
            case ast_.Quantifier(elem, expression, op_type):
                new_quantifier = ast_.Quantifier(
                    ast_.And([elem]),
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

                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in quantifier (cannot choose generators)")
                    return None

                if all(map(lambda x: isinstance(x, GeneratorSelection) or isinstance(x, CombinedGeneratorSelection), elems)):
                    logger.debug(f"FAILED: all elements in OR quantifier are already GeneratorSelections, no need to select generators")
                    return None

                gsps: list[GeneratorSelection] = []
                for elem in elems:
                    if not isinstance(elem, ast_.ListOp) or elem.op_type != ast_.ListOperator.AND:
                        logger.debug(f"FAILED: element in OR quantifier is not an And operation (got {elem})")
                        return None

                    gsp = self._make_gsp_from_and_clause(elem.items, ast._bound_identifiers)
                    if gsp is None:
                        logger.debug(f"FAILED: could not make GSP from and clause {elem} with bound identifiers {ast._bound_identifiers}")
                        return None

                    gsps.append(gsp)

                if not gsps:
                    logger.debug(f"FAILED: no valid generator selections found in OR quantifier (got {elems})")
                    return None

                # No _bound_identifiers from this point on?
                ret = ast_.Quantifier(
                    ast_.Or(elems),
                    expression,
                    op_type,
                )
                ret._bound_identifiers = ast._bound_identifiers
                return ret
                # predicates: list[ast_.ASTNode] = []
                # generators_with_alternatives: list[list[ast_.In]] = []
                # for elem in elems:

            # This should only match once per quantifier
            # For each And inside the Or:
            # - Only match if valid generator selection:
            #   - Generator structure exists with LHS in _bound_identifiers
            # - Then:
            #   - Only one generator per bound identifier - need to find all alternatives and then restrict from there
            #   - Try to order in list as a composition chain
        return None

    def _make_gsp_from_and_clause(
        self,
        and_clause: list[ast_.ASTNode],
        bound_identifiers: set[ast_.Identifier | ast_.MapletIdentifier],
    ) -> GeneratorSelection | None:
        candidate_generators_per_identifier: dict[ast_.Identifier | ast_.MapletIdentifier, set[ast_.In]] = {identifier: set() for identifier in bound_identifiers}
        other_predicates: list[ast_.ASTNode] = []
        for elem in and_clause:
            if not isinstance(elem, ast_.BinaryOp):
                other_predicates.append(elem)
                continue
            if any(
                [
                    elem.op_type != ast_.BinaryOperator.IN,
                    not isinstance(elem.left, ast_.Identifier | ast_.MapletIdentifier),
                    elem.left not in candidate_generators_per_identifier,
                    isinstance(elem.right.get_type, ast_.SetType),
                ]
            ):
                other_predicates.append(elem)
                continue

            assert isinstance(elem.left, ast_.Identifier | ast_.MapletIdentifier)
            casted_elem = ast_.In(elem.left, elem.right)
            candidate_generators_per_identifier[elem.left].add(casted_elem)

        # Ensure every identifier has at least one suitable generator
        for identifier, candidate_generators in candidate_generators_per_identifier.items():
            if len(candidate_generators) == 0:
                logger.debug(f"Failed to find a suitable generator for identifier {identifier}. Dict of generators: {candidate_generators_per_identifier}")
                return None

        identifiers: set[ast_.Identifier] = set(filter(lambda x: isinstance(x, ast_.Identifier), bound_identifiers))  # type: ignore
        # There may be more than one chain, chains will contain a sequence of MapletIdentifiers
        maplets: set[ast_.MapletIdentifier] = set(filter(lambda x: isinstance(x, ast_.MapletIdentifier), bound_identifiers))  # type: ignore
        maplet_chains: list[list[ast_.MapletIdentifier]] = []

        while maplets:
            maplet = maplets.pop()
            # Each maplet should only occur once, and only be added to a chain once (variable names are guaranteed to be unique)

            for i in range(len(maplet_chains)):
                if any(
                    [
                        maplet.left == maplet_chains[i][-1].right,
                        ast_.Equal(maplet.left, maplet_chains[i][-1].right) in other_predicates,
                        ast_.Equal(maplet_chains[i][-1].right, maplet.left) in other_predicates,
                    ]
                ):
                    maplet_chains[i].append(maplet)
                    break

                if any(
                    [
                        maplet.right == maplet_chains[i][0].left,
                        ast_.Equal(maplet.right, maplet_chains[i][0].left) in other_predicates,
                        ast_.Equal(maplet_chains[i][0].left, maplet.right) in other_predicates,
                    ]
                ):
                    maplet_chains[i].insert(0, maplet)
                    break
            else:
                maplet_chains.append([maplet])

        # TODO make a proper ordering for this, base on size, relation subtype, etc.
        generators = []
        # For now, take the longest chain first
        sorted_maplet_chains = sorted(maplet_chains, key=lambda chain: len(chain), reverse=True)
        for maplet_chain in sorted_maplet_chains:
            for maplet in maplet_chain:
                generators.append(candidate_generators_per_identifier[maplet].pop())
        for identifier in identifiers:
            generators.append(candidate_generators_per_identifier[identifier].pop())

        for unselected_candidate_generators in candidate_generators_per_identifier.values():
            other_predicates.extend(list(unselected_candidate_generators))

        return GeneratorSelection(generators, ast_.And(other_predicates))

    def reduce_duplicate_generators(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.OR):
                if all(map(lambda x: isinstance(x, GeneratorSelection) or isinstance(x, CombinedGeneratorSelection), elems)):
                    logger.debug(f"FAILED: all elements in OR quantifier are already GeneratorSelections, no need to select generators")
                    return None

                combination_dict: dict[ast_.In, list[GeneratorSelection | CombinedGeneratorSelection]] = {}
                for elem in elems:
                    match elem:
                        case GeneratorSelection(generators, predicates):
                            if not combination_dict.get(generators[0]):
                                combination_dict[generators[0]] = []
                            combination_dict[generators[0]].append(
                                GeneratorSelection(
                                    generators[1:],
                                    predicates,
                                )
                            )
                        case CombinedGeneratorSelection(generator, child_generators):
                            if not combination_dict.get(generator):
                                combination_dict[generator] = []
                            combination_dict[generator].extend(child_generators.items)  # type: ignore
                        case _:
                            logger.debug(f"FAILED: element is not a valid GeneratorSelection or CombinedGeneratorSelection (got {elem})")
                            return None

                if len(elems) != len([y for x in combination_dict.values() for y in x]):
                    logger.debug("FAILED: some elements were not added to combination dict - this should not happen")
                    return None

                combined_generators: list[ast_.ASTNode] = []
                for combined_generator, generator_list in combination_dict.items():
                    assert isinstance(combined_generator.left, ast_.Identifier | ast_.MapletIdentifier), "Combined generator should have an identifier on the left side"

                    # If the generator list only has one entry, we didn't combine anything - recreate the original GeneratorSelectionV2
                    if len(generator_list) == 1:
                        combined_generators.append(
                            GeneratorSelection(
                                [combined_generator],
                                generator_list[0].predicates,
                            )
                        )
                        continue

                    # Actually combine generators that need to be combined. Note that this may end up
                    # creating GeneratorSelections without a generator.
                    combined_generators.append(
                        CombinedGeneratorSelection(
                            combined_generator,
                            ast_.Or(generator_list),  # type: ignore
                        )
                    )

                return ast_.Or(combined_generators)

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
                if not all(map(lambda x: isinstance(x, GeneratorSelection) or isinstance(x, CombinedGeneratorSelection), predicate.items)):
                    logger.debug(f"FAILED: not all items in predicate are GeneratorSelections (got {predicate.items}). Elements of predicates should be GeneratorSelectionASTs")
                    return None

                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in quantifier (cannot perform optimizations)")
                    return None

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
                ast_.ListOp(generators, ast_.ListOperator.OR),
                body,
            ):
                if not all(map(lambda x: isinstance(x, GeneratorSelection) or isinstance(x, CombinedGeneratorSelection), generators)):
                    logger.debug(f"FAILED: all elements in top level OR predicate expected to be GeneratorSelections (got {generators}).")
                    return None

                statements: list[ast_.ASTNode] = []
                used_generators: list[ast_.ASTNode] = []
                for generator in generators:
                    assert isinstance(generator, GeneratorSelection | CombinedGeneratorSelection)
                    generator.predicates = ast_.And([generator.predicates, ast_.Not(ast_.Or(used_generators))])

                    statements.append(Loop(generator, deepcopy(body)))
                    used_generators.append(generator.flatten())

                return ast_.Statements(statements)

        return None

    def chained_gsp_loop(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            # Inline empty_gsp_loop rule
            case Loop(
                GeneratorSelection([], predicates),
                body,
            ):
                return ast_.If(predicates, body)
            case Loop(
                GeneratorSelection(generators, predicates),
                body,
            ):
                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in loop (cannot perform optimizations)")
                    return None

                free_predicates = []
                bound_predicates = []
                identifiers_within_child_generators: list[ast_.Identifier] = ast_.flatten(map(lambda x: x.left.find_all_instances(ast_.Identifier), generators[1:]))
                for predicate in predicates.items:
                    identifiers_within_predicate = predicate.find_all_instances(ast_.Identifier)

                    for identifier in identifiers_within_predicate:
                        if (
                            # If the identifier is not bound outside of the quantifier (and is not the current bound quantifier var)
                            ast._env.get(identifier.name) is None
                            and identifier not in generators[0].left.find_all_instances(ast_.Identifier)
                            # or if it is used within a child generator
                        ) or identifier in identifiers_within_child_generators:
                            # propagate predicate to child
                            bound_predicates.append(predicate)
                            break
                    else:
                        free_predicates.append(predicate)

                assert isinstance(generators[0].left, ast_.Identifier | ast_.MapletIdentifier), "Generators should have an identifier on the left side"
                return Loop(
                    SingleGeneratorSelection(
                        generators[0],
                        ast_.And(
                            free_predicates,
                        ),
                    ),
                    Loop(
                        GeneratorSelection(
                            generators[1:],
                            ast_.And(bound_predicates),
                        ),
                        body,
                    ),
                )

        return None

    def combined_gsp_loop(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                CombinedGeneratorSelection(
                    generator,
                    gsp_predicates,
                    predicates,
                ),
                body,
            ):
                child_loops: list[ast_.ASTNode] = []
                for gsp_predicate in gsp_predicates.items:
                    if not isinstance(gsp_predicate, GeneratorSelection | CombinedGeneratorSelection | SingleGeneratorSelection):
                        logger.debug(f"FAILED: gsp predicate is not a valid GeneratorSelection (got {gsp_predicate})")
                        return None

                    child_loops.append(Loop(gsp_predicate, body))

                return Loop(
                    SingleGeneratorSelection(generator, predicates),
                    ast_.Statements(child_loops),
                )
        return None


class LoopsCodeGenerationCollection(RewriteCollection):

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.conjunct_conditional,
        ]

    def conjunct_conditional(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case Loop(
                SingleGeneratorSelection(
                    generator,
                    predicates,
                ),
                body,
            ):
                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in loop (cannot perform optimizations)")
                    return None

                free_predicates = []
                bound_predicates = []
                for predicate in predicates.items:
                    identifiers_within_predicate = predicate.find_all_instances(ast_.Identifier)

                    for identifier in identifiers_within_predicate:
                        # If the identifier is not bound outside of the quantifier or if it is used within the generator
                        if ast._env.get(identifier.name) is None or generator.contains_item(identifier):
                            bound_predicates.append(predicate)
                            break
                    else:
                        free_predicates.append(predicate)

                statement: ast_.ASTNode = body
                if bound_predicates:
                    statement = ast_.Statements(
                        [
                            ast_.If(
                                ast_.And(bound_predicates),
                                statement,
                            )
                        ]
                    )

                assert isinstance(generator.left, ast_.Identifier | ast_.MapletIdentifier), f"Generator should have an identifier on the left side (got {generator.left})"
                statement = ast_.For(
                    ast_.IdentList([generator.left]),
                    generator.right,
                    statement,
                )

                if free_predicates:
                    statement = ast_.If(
                        ast_.And(free_predicates),
                        body=ast_.Statements([statement]),
                    )

                return ast_.Statements([statement])

        return None


class ReplaceAndSimplifyCollection(RewriteCollection):
    bound_generator_variables: set[ast_.Identifier] = field(default_factory=set)

    def apply_all_rules_one_traversal(self, ast):
        # Before entering a new quantifier, record currently bound variables
        # (so we can restore them later)
        bound_generator_variables_before = deepcopy(self.bound_generator_variables)
        if isinstance(ast, ast_.For):
            logger.debug(f"For loop found with bound variables: {ast.iterable_names.flatten()}")
            # Add newly accessible (bound) loop variables, used for equality elimination
            self.bound_generator_variables |= ast.iterable_names.flatten()

        ast = super().apply_all_rules_one_traversal(ast)
        logger.debug(f"AST after applying all rules: {ast.pretty_print_algorithmic()}")

        # Restore bound variable record since we have exited the possibly nested quantifier
        self.bound_generator_variables = bound_generator_variables_before

        return ast

    def _rewrite_collection(self) -> list[Callable[[ast_.ASTNode], ast_.ASTNode | None]]:
        return [
            self.equality_elimination,
            self.simplify_equalities,
            self.simplify_and,
            self.simplify_or,
            self.flatten_nested_statements,
        ]

    def equality_elimination(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.If(
                ast_.ListOp(elems, ast_.ListOperator.AND) as predicate,
                body,
            ):
                if ast._env is None:
                    logger.debug(f"FAILED: no environment found in if (cannot perform optimizations)")
                    return None

                new_predicate_items = predicate.items
                substitution = None
                for and_clause in elems:
                    if not isinstance(and_clause, ast_.BinaryOp) or and_clause.op_type != ast_.BinaryOperator.EQUAL:
                        continue

                    if isinstance(and_clause.left, ast_.Identifier) and and_clause.left not in self.bound_generator_variables and ast._env.get(and_clause.left.name) is None:
                        substitution = and_clause
                        new_predicate_items.remove(and_clause)
                        break
                    if isinstance(and_clause.right, ast_.Identifier) and and_clause.right not in self.bound_generator_variables and ast._env.get(and_clause.right.name) is None:
                        substitution = ast_.Equal(
                            and_clause.right,
                            and_clause.left,
                        )
                        new_predicate_items.remove(and_clause)
                        break

                if not substitution:
                    logger.debug(f"FAILED: no substitutions found in OR quantifier (current environment is {ast._env}), free variables are {ast.free}")
                    return None
                logger.debug(f"Running substitution {substitution} in {elems}")

                new_predicate = ast_.And(new_predicate_items)
                new_predicate.find_and_replace(
                    substitution.left,
                    substitution.right,
                )
                body.find_and_replace(
                    substitution.left,
                    substitution.right,
                )
                return ast_.If(
                    new_predicate,
                    body,
                )

        return None

    def simplify_equalities(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
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

                return ast_.And(new_elems)
        return None

    def simplify_or(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.ListOp(elems, ast_.ListOperator.OR):
                new_elems = []
                for elem in elems:
                    if isinstance(elem, ast_.True_):
                        return ast_.True_()
                    if isinstance(elem, ast_.False_):
                        continue
                    new_elems.append(elem)

                if elems == new_elems:
                    logger.debug("FAILED: no simplification applied to OR list operation (no clauses were removed)")
                    return None

                if not new_elems:
                    return ast_.Or([])

                return ast_.Or(new_elems)
        return None

    def flatten_nested_statements(self, ast: ast_.ASTNode) -> ast_.ASTNode | None:
        match ast:
            case ast_.Statements(items):

                if not any(map(lambda x: isinstance(x, ast_.Statements), items)):
                    return None

                new_statements: list[ast_.ASTNode] = []
                for item in items:
                    if not isinstance(item, ast_.Statements):
                        new_statements.append(item)
                        continue

                    new_statements.extend(item.items)

                return ast_.Statements(new_statements)
        return None


REWRITE_COLLECTION: list[type[RewriteCollection]] = [
    SyntacticSugarForBags,
    BuiltinFunctions,
    ComprehensionConstructionCollection,
    DisjunctiveNormalFormCollection,
    OrWrappingCollection,
    GeneratorSelectionCollection,
    GSPToLoopsCollection,
    LoopsCodeGenerationCollection,
    ReplaceAndSimplifyCollection,
]
