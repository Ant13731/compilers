from dataclasses import dataclass, field
from uuid import uuid4
from typing import ClassVar, Callable, Self

try:
    from . import ast_  # type: ignore
except ImportError:
    import ast_  # type: ignore


# Following Baader
@dataclass
class RuleVar(ast_.ASTNode):
    """For defining variables in matching/rewrite rules"""

    id: str

    # uuid: str = field(default_factory=lambda: str(uuid4()), init=False)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuleVar):
            return False
        return self.id == other.id


Substitution = dict[RuleVar, ast_.ASTNode]
PreMatchedASTNode = ast_.ASTNode


@dataclass
class RewriteRule:
    lh: ast_.ASTNode
    rh: ast_.ASTNode
    match_condition: Callable[[Self, PreMatchedASTNode], bool] | None = None
    substitution_condition: Callable[[Self, Substitution, PreMatchedASTNode], bool] | None = None


# def indom(v: RuleVar, s: Substitution) -> bool:
#     return any(map(lambda y: v == y, s))


# def app(s: Substitution, v: RuleVar) -> ast_.ASTNode | None:
#     return s.get(v)


# def lift(s: Substitution, t: ast_.ASTNode) -> ast_.ASTNode:
#     if isinstance(t, RuleVar):
#         if indom(t, s):
#             return app(s, t)
#         else:
#             return t
#     else:
#         return t.__class__(
#             *[lift(s, f) for f in t.fields()],
#         )


# Trying other stuff
x = RuleVar("x")
rule_example: RewriteRule = RewriteRule(
    ast_.BinaryOp(x, x, op_type=ast_.BinaryOpType.MULTIPLY),
    ast_.BinaryOp(x, ast_.Int("2"), op_type=ast_.BinaryOpType.EXPONENT),
)


# def match(lh: ast_.ASTNode, ast: ast_.ASTNode) -> Substitution:
#     match_list = [(lh, ast)]
#     return match_aux(match_list, {})


# class UnifyException(Exception):
#     pass


# cases for match_aux:
# lh = RuleVar, ast = Term
# lh = Term, ast = RuleVar
# lh = Term, ast = Term
#  - try to match children in order (may be tricky bc we use builtin lists...)
#  - idea: use dataclass_traverse to get children in a specific order
#
# then when writing out rules, make sure each match gets a separate rulevar identifier


# # trying the book stuff, likely need to change into more imperative style...
# # term is built very differently,,, (based off of page 89 of baader)
# def match_aux(match_list: list[tuple[ast_.ASTNode, ast_.ASTNode]], s: Substitution) -> Substitution:
#     if not match_list:
#         return s

#     lh, ast = match_list.pop(0)
#     if isinstance(lh, RuleVar):
#         if lh in s:
#             if s[lh] == ast:
#                 return match_aux(match_list, s)
#             else:
#                 raise UnifyException
#         else:
#             return match_aux(match_list, {**s, lh: ast})
#     if isinstance(ast, RuleVar):
#         raise UnifyException
#     if type(lh) != type(ast):
#         raise UnifyException
#     return match_aux(list(zip(lh.list_children(), ast.list_children())), s)


def match(lh: ast_.ASTNode, ast: ast_.ASTNode) -> Substitution | None:
    match_list = [(lh, ast)]
    substitutions = {}
    while match_list:
        lht, t = match_list.pop(0)
        if isinstance(lht, RuleVar):
            # Current t is now a substitution candidate

            # Add to substitution map if entry does not exist
            if lht not in substitutions:
                substitutions[lht] = t
                continue

            # Check if it already exists in the substitution map
            if substitutions.get(lht) == t:
                continue

            # Otherwise, mismatch => failed to match
            return None

        if ast_.is_dataclass_leaf(t):
            if lht == t:
                continue
            # cant match to a variable - will ast ever have a variable tho??
            # maybe we need to replace this with the ast leaf nodes (like int, float, etc.)
            #
            # how would we match something like lh=Int(2) and ast=Int(2)
            # We may need something to match literals?
            #
            # Idea, values with no ASTNode children can be matched directly
            # Candidate leaf nodes:
            # Int, Float, String, True_, False_, None_, Identifier?
            # ControlFlowStmt, ImportAll
            else:
                return None

        # At this point, both lht and t are AST nodes (terms) of the same type
        if type(lht) != type(t):
            return None

        match_list += list(zip(lht.children(), t.children()))

    return substitutions


def substitute(rh: ast_.ASTNode, s: Substitution) -> ast_.ASTNode:
    if isinstance(rh, RuleVar):
        return s.get(rh, rh)  # If not in substitution, return the variable itself
    if ast_.is_dataclass_leaf(rh):
        return rh
    # Rebuild RH with substituted children
    return rh.__class__(
        *[substitute(f, s) for f in rh.children()],
    )


class MatchingPhase:
    """Each matching phase contains a list of rules that are applied to the AST in order exhaustively, starting from the root.

    Exit conditions separate these phases, allowing for early termination of the matching process."""

    rules: ClassVar[list[RewriteRule]]

    # @classmethod
    # def rewrite(cls, ast: ast_.ASTNode) -> ast_.ASTNode | None:
    #     raise NotImplementedError

    @classmethod
    def exit_condition(cls, ast: ast_.ASTNode) -> bool:
        raise NotImplementedError


def helper_substitution_condition(self: RewriteRule, substitution: Substitution, ast: ast_.ASTNode) -> bool:
    sub = substitution.get(RuleVar("x"), None)
    if isinstance(sub, ast_.Int):
        substitution[RuleVar("x_modified")] = ast_.Int(str(int(sub.value) * 2))
        return True
    return False


class TestMatchingPhase(MatchingPhase):
    rules: ClassVar[list[RewriteRule]] = [
        RewriteRule(
            ast_.BinaryOp(RuleVar("x"), RuleVar("x"), op_type=ast_.BinaryOpType.MULTIPLY),
            ast_.BinaryOp(RuleVar("x"), ast_.Int("2"), op_type=ast_.BinaryOpType.EXPONENT),
        ),
        RewriteRule(
            ast_.BinaryOp(RuleVar("x"), RuleVar("x"), op_type=ast_.BinaryOpType.ADD),
            ast_.BinaryOp(RuleVar("x"), ast_.Int("2"), op_type=ast_.BinaryOpType.MULTIPLY),
        ),
        # RewriteRule(
        #     ast_.BinaryOp(RuleVar("x"), ast_.Int("2"), op_type=ast_.BinaryOpType.MULTIPLY),
        #     RuleVar("x_modified"),
        #     substitution_condition=helper_substitution_condition,
        # ),
    ]

    @classmethod
    def exit_condition(cls, ast: ast_.ASTNode) -> bool:
        return False  # No exit condition for testing purposes


# class PhaseOne(MatchingPhase):
#     @classmethod
#     def rewrite(cls, ast: ast_.ASTNode) -> ast_.ASTNode | None:
#         match ast:
#             case ast_.BinaryOp(x, y, op_type=ast_.BinaryOpType.MULTIPLY) if x == y:
#                 return ast_.BinaryOp(x, ast_.Int("2"), op_type=ast_.BinaryOpType.EXPONENT)
#             case _:
#                 return None


def apply_all_rules_once(ast: ast_.ASTNode, matching_phase: type[MatchingPhase]) -> ast_.ASTNode:
    for rewrite_rule in matching_phase.rules:
        if matching_phase.exit_condition(ast):
            print(f"Exit condition met for phase {matching_phase.__class__.__name__}, stopping rule application.")
            return ast

        # First check match condition to determine if we want to attempt rule application
        if rewrite_rule.match_condition is not None and not rewrite_rule.match_condition(ast):
            continue

        print(f"ATTEMPT: to match rule: {rewrite_rule.lh} -> {rewrite_rule.rh} with AST: {ast}")

        # Attempt to match the left-hand side (lh) with the AST (ast)
        substitutions = match(rewrite_rule.lh, ast)

        if substitutions is None:
            print(f"FAILED: to match lh side of rule: {rewrite_rule.lh} with AST: {ast}")
            continue

        if rewrite_rule.substitution_condition is not None and not rewrite_rule.substitution_condition(rewrite_rule, substitutions, ast):
            print(f"FAILED: substitution condition for rule {rewrite_rule.rh} with substitutions {substitutions}")
            continue

        # If a match is found, rewrite the AST with the right-hand side (rh)
        ast = substitute(rewrite_rule.rh, substitutions)
        print(f"SUCCESS: applied substitutions {substitutions} to rule {rewrite_rule.rh}, resulting in new ast: {ast}")
    return ast


def normalizer(ast: ast_.ASTNode, matching_phase: type[MatchingPhase]) -> ast_.ASTNode:

    new_ast_children = []
    for child in ast.children():
        if not isinstance(child, ast_.ASTNode):
            new_ast_children.append(child)
            continue
        # Recursively normalize children first
        child = normalizer(child, matching_phase)
        new_ast_children.append(child)

    ast = ast.__class__(*new_ast_children)
    print(f"\nNormalizing AST: {ast}")
    return apply_all_rules_once(ast, matching_phase)

    # if ast.is_leaf():
    #     return ast
    # u = ast.__class__(
    #     *[normalizer(f, matching_phase) for f in ast.children()],
    # )
    # return normalizer(apply_all_rules_once(u, matching_phase), matching_phase)


def collection_optimizer(ast: ast_.ASTNode, matching_phases: list[type[MatchingPhase]]) -> ast_.ASTNode:
    for matching_phase in matching_phases:
        print(f"Applying matching phase: {matching_phase.__name__}")
        ast = normalizer(ast, matching_phase)
    return ast


def test() -> None:
    # Example usage
    print("Starting collection optimizer test...")
    practice_ast: ast_.ASTNode = ast_.BinaryOp(
        ast_.BinaryOp(
            ast_.Int("2"),
            ast_.Int("2"),
            op_type=ast_.BinaryOpType.ADD,
        ),
        ast_.Int("4"),
        op_type=ast_.BinaryOpType.MULTIPLY,
    )
    print("Original AST:", practice_ast.pretty_print())
    optimized_ast = collection_optimizer(practice_ast, [TestMatchingPhase])
    print("Optimized AST:", optimized_ast.pretty_print())


test()
