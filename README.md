# Compilers

Welcome to the very high level language repo! The repository is organized as follows:

- `directed_readings`: Reports and assignments for the COMPSCI 4Z03 Directed Readings course.
  - `ADT`: A Survey of Abstract Data Types (in several languages)
  - `implementation`: A report for the first version of a prototype compiler
  - `implementation_v2`: Revised report for a prototype compiler
  - `term_rewriting`: A Survey of Term Rewriting for Compiler Optimization
- `implementation`: Source code for prototypical compiler
  - `rewrite_strategy_select_examples`: Collection of files used in `rewrite_strategy_select_examples_test.py`
    - `set_strategy.py`: A simplified implementation of the rewrite rule process, as described in Section 3.2 of `directed_readings/implementation_v2/report.pdf`. The current implementation does not handle conditions on the properties of dummy variables, multiple dummy variables, or functions on the dummy variable. Instead, it offers basic rewrite rules for denesting, union, intersection, and set difference.
    - `test_items.py`: Ground truth objects to test the rewrite rule set against. Generally, the process goes from plaintext -> Simple AST -> Verbose AST (where sets are all in constructor form) -> Verbose AST with no set operations (only boolean operations) -> Concrete AST -> Code output.
  - `ast_.py`: Collection of (empty-for-now) classes to represent the AST of our very high level language
  - `grammar.lark`: A CFG for our very high level language, written in a dialect suitable for the Lark Python library
  - `parser.py`: Contains a function that accepts a plaintext language input and returns an AST if valid in the language.
  - `README.md`: Progress tracker for the language
  - `rewrite_strategy.py`: Plan for rewrite pass of the optimizing compiler. Currently not implemented.
  - `rewrite_strategy_select_examples_test.py`: Select examples of rewrite strategy implementations, independent from the rest of the compiler. This test suite relies on `implementation/rewrite_strategy_select_examples`
  - `transformer.py`: An API into lark that converts Lark Tree tokens into our custom AST classes.
  - `transformer_unit_test.py`: Unit tests for transformer.py
- `proposals`: Proposals discussing the problems and justification of our work, WIP

# Important dates

- OGS - May 14
- SYNT 2025 - May 18
- CASCON 2025 - May 19
- NSERC - Dec 1
- Google - Apr 30 (next year I assume)

# TODO (non-implementation)
