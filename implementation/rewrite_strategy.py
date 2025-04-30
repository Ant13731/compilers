from ast_ import *


class RewriteStrategyBase:
    """
    Base class for rewrite strategies.
    """

    def rewrite(self, ast: BaseAST) -> BaseAST:
        """
        Rewrite the given AST using the strategy.
        """
        raise NotImplementedError("Rewrite method not implemented in base class.")


class StageOneSetStrategy(RewriteStrategyBase):
    """
    Stage one rewrite strategy for converting all Sets into constructor notation.

    Intended for use with other stages as listed in Section 3.2 of the Implementation paper.
    """

    @staticmethod
    def rewrite_visitor(ast: BaseAST) -> BaseAST:
        pass
        # TODO tomorrow
        # - add type information to variables, both where they are defined and used.
        # - eventually every class will need to have an associated type, particularly operators. use None type for things that have no value, like statements
        # - maybe a separate pass for the context too? that way we could, look up the type or value of an identifier, for example?
        # maybe we will need a middle pass to find and attach that information?
        # Need to differentiate where they are defined and used, only convert used variables set notation here

    #     match ast:
    #         case Identifier(value):

    # def rewrite(self, ast: BaseAST) -> BaseAST:
