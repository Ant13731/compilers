from dataclasses import dataclass, field
from uuid import uuid4
from typing import ClassVar

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
RewriteRule = tuple[ast_.ASTNode, ast_.ASTNode]  # (pattern, replacement)


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
rule_example: RewriteRule = (
    ast_.BinaryOp(x, x, op_type=ast_.BinaryOpType.MULTIPLY),
    ast_.BinaryOp(x, ast_.Int("2"), op_type=ast_.BinaryOpType.EXPONENT),
)

practice_ast: ast_.ASTNode = ast_.BinaryOp(
    ast_.BinaryOp(
        ast_.Int("2"),
        ast_.Int("2"),
        op_type=ast_.BinaryOpType.ADD,
    ),
    ast_.Int("4"),
    op_type=ast_.BinaryOpType.MULTIPLY,
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

        if isinstance(t, RuleVar):
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
            return None

        # At this point, both lht and t are AST nodes (terms) of the same type
        if type(lht) != type(t):
            return None

        match_list += list(zip(lht.list_children(), t.list_children()))

    return substitutions


def substitute(rh: ast_.ASTNode, s: Substitution) -> ast_.ASTNode:
    if isinstance(rh, RuleVar):
        return s.get(rh, rh)  # If not in substitution, return the variable itself
    # Rebuild RH with substituted children
    return rh.__class__(
        *[substitute(f, s) for f in rh.list_children()],
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


# class PhaseOne(MatchingPhase):
#     @classmethod
#     def rewrite(cls, ast: ast_.ASTNode) -> ast_.ASTNode | None:
#         match ast:
#             case ast_.BinaryOp(x, y, op_type=ast_.BinaryOpType.MULTIPLY) if x == y:
#                 return ast_.BinaryOp(x, ast_.Int("2"), op_type=ast_.BinaryOpType.EXPONENT)
#             case _:
#                 return None


def collection_optimizer(ast: ast_.ASTNode, matching_phases: list[MatchingPhase]) -> ast_.ASTNode:
    for matching_phase in matching_phases:
        while not matching_phase.exit_condition(ast):
            continue
            # ast = ast_.find_and_replace(ast, matching_phase.rewrite)

        # for pattern, replacement in matching_phase.rules:
        #     # Apply the pattern to the AST and replace it with the replacement
        #     # This is a placeholder for the actual matching logic
        #     # if ast.contains(pattern):
        #     #     ast = ast.replace(pattern, replacement)
        #     # TODO attempt match and then rewrite (should take in an ast and return the modified ast if a match is found)
        #     if matching_phase.exit_condition(ast):
        #         break

    return ast
