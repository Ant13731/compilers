from src.mod.pipeline.optimizer.v2.simrw import GuardCondition, RewriteRule
from src.mod.pipeline.printers.type_ import type_to_source
from src.mod.pipeline.printers.ast_ import ast_to_source


def simrw_to_source(rewrite_rule: RewriteRule) -> str:
    ret = f"rule {rewrite_rule.name}:\n"
    ret += "\tvars:\n"
    if rewrite_rule.vars_:
        ret += "\t\t"
        ret += "\n\t\t".join(f"{var}: {type_to_source(type_)}" for var, type_ in rewrite_rule.vars_.items())
        ret += "\n"
    ret += f"\trewrite:\n"
    ret += f"\t\t{ast_to_source(rewrite_rule.rewrite_left)}\n"
    ret += "\t\t~>\n"
    ret += f"\t\t{ast_to_source(rewrite_rule.rewrite_right)}\n"
    if rewrite_rule.when:
        ret += "\twhen:\n"
        ret += "\t\t"
        ret += "\n\t\t".join(f"{guard_condition_to_source(condition)}" for condition in rewrite_rule.when)
        ret += "\n"
    return ret


def guard_condition_to_source(guard_condition: GuardCondition) -> str:
    raise NotImplementedError  # TODO need to implement visitor pattern over all variable types of guards...
    return f"{guard_condition.name}()"
