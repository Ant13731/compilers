from functools import singledispatch
from dataclasses import dataclass

from llvmlite import ir

from src.mod import ast_

# Types
Int = ir.IntType(64)
Float = ir.DoubleType()


@dataclass
class CodeGenerationContext:
    pass


@singledispatch
def generate_llvm_code(node: ast_.ASTNode, context: CodeGenerationContext):
    """
    Generate LLVM IR code for a given node in the AST.

    Args:
        node: The AST node to generate code for.
        context: The context containing information about the current compilation state.

    Returns:
        An LLVM IR value representing the generated code.
    """
    raise NotImplementedError(f"Code generation not implemented for node type: {type(node)}")


@generate_llvm_code.register
def _(node: ast_.Identifier, context: CodeGenerationContext): ...
