The goal of using LLVM/MLIR as the backend target its to gain the benefits of a low level optimizer without needing to focus on those optimizations. To access MLIR's suite of optimizations, we must lower Simile's AST into MLIR format

# Learning
## MLIR
- few pre-defined instructions/operations/types
    - focus on dialects
- Dialect
    - define new operations, attributes (metadata), and types
    - has a unique namespace
- Operation
    - core unit of abstraction/computation, like an assembly instruction
    - Contain:
        - name
        - list of SSA operand values
        - list of attributes
            - dictionary for constant data, MLIR provides builtin values similar to json
        - list of result types
        - source location for debugging - must remain attached through optimization
        - successor blocks
        - regions (list of blocks)
    - all customizable elements can be reduced to these parts
        - opaque API, minimal constraints