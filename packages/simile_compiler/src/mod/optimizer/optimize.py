from src.mod.config import debug_print
from src.mod import ast_
from src.mod.optimizer.base_types import RuleVar, Substitution, MatchingPhase


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


def apply_all_rules_once(ast: ast_.ASTNode, matching_phase: type[MatchingPhase]) -> ast_.ASTNode:
    for rewrite_rule in matching_phase.rules:
        if matching_phase.exit_condition(ast):
            debug_print(f"Exit condition met for phase {matching_phase.__class__.__name__}, stopping rule application.")
            return ast

        # First check match condition to determine if we want to attempt rule application
        if rewrite_rule.match_condition is not None and not rewrite_rule.match_condition(ast):  # type: ignore
            continue

        debug_print(f"ATTEMPT: to match rule: {rewrite_rule.lh} -> {rewrite_rule.rh} with AST: {ast}")

        # Attempt to match the left-hand side (lh) with the AST (ast)
        substitutions = match(rewrite_rule.lh, ast)

        if substitutions is None:
            debug_print(f"FAILED: to match lh side of rule: {rewrite_rule.lh} with AST: {ast}")
            continue

        if rewrite_rule.substitution_condition is not None and not rewrite_rule.substitution_condition(rewrite_rule, substitutions, ast):
            debug_print(f"FAILED: substitution condition for rule {rewrite_rule.rh} with substitutions {substitutions}")
            continue

        # If a match is found, rewrite the AST with the right-hand side (rh)
        ast = substitute(rewrite_rule.rh, substitutions)
        debug_print(f"SUCCESS: applied substitutions {substitutions} to rule {rewrite_rule.rh}, resulting in new ast: {ast}")
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
    debug_print(f"\nNormalizing AST: {ast}")
    return apply_all_rules_once(ast, matching_phase)


def collection_optimizer(ast: ast_.ASTNode, matching_phases: list[type[MatchingPhase]]) -> ast_.ASTNode:
    for matching_phase in matching_phases:
        debug_print(f"Applying matching phase: {matching_phase.__name__}")
        ast = normalizer(ast, matching_phase)
    return ast
