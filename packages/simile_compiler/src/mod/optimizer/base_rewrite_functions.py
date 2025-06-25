from src.mod import ast_
from src.mod.optimizer.base_types import RuleVar, Substitution


def match(lh: ast_.ASTNode, ast: ast_.ASTNode) -> Substitution | None:
    """Attempts to match the left-hand side (lh) of one rule with the given AST node (ast).
    On success, returns a substitution map that can be used with the right-hand side to rewrite the AST.
    """
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
    """Substitutes the right-hand side (rh) of a rule with the given substitution map (s)."""
    if isinstance(rh, RuleVar):
        return s.get(rh, rh)  # If not in substitution, return the variable itself
    if ast_.is_dataclass_leaf(rh):
        return rh
    # Rebuild RH with substituted children
    return rh.__class__(
        *[substitute(f, s) for f in rh.children()],
    )
