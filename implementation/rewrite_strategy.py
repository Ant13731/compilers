from ast_ import *


# NOTE: Currently unused, planning for future organization of rewrite strategies
class RewritingStrategies:
    """
    Not so much a class, but a collection of rewriting strategies for transforming our language ASTs.
    """

    @staticmethod
    def set_rewrite(ast: BaseAST) -> BaseAST:
        """Rewrite strategy for all set-based constructs.

        Follows the multi-stage outline as listed in Section 3.2 of the Implementation paper."""
        ast = RewritingStrategies.set_rewrite_stage_1(ast)
        ast = RewritingStrategies.set_rewrite_stage_2(ast)
        ast = RewritingStrategies.set_rewrite_stage_3(ast)
        return ast

    @staticmethod
    def set_rewrite_stage_1(ast: BaseAST) -> BaseAST:
        """
        Stage one: Convert all Sets into constructor notation.
        """
        raise NotImplementedError("Stage 1 not implemented yet.")

    @staticmethod
    def set_rewrite_stage_2(ast: BaseAST) -> BaseAST:
        """
        Stage two: Convert set operators into boolean-based set constructions.
        """
        raise NotImplementedError("Stage 2 not implemented yet.")

    @staticmethod
    def set_rewrite_stage_3(ast: BaseAST) -> BaseAST:
        """
        Stage three: Logical condition manipulation. Selects "generators" for each set.
        """
        raise NotImplementedError("Stage 3 not implemented yet.")
