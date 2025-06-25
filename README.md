# Compilers

Welcome to the very high level language repo! The repository is organized as follows:

- `docs`: Reports and assignments for the COMPSCI 4Z03 Directed Readings course.
  - `ADT`: A Survey of Abstract Data Types (in several languages)
  - `implementation`: A report for the first version of a prototype compiler
  - `implementation_v2`: Revised report for a prototype compiler
  - `term_rewriting`: A Survey of Term Rewriting for Compiler Optimization
  - `trs_specification`: Complete TRS Specification for Abstract Collection Types
- `implementation`: Source code for prototypical compiler
  - `rewrite_strategy_select_examples`: Collection of files used in `rewrite_strategy_select_examples_test.py`
    - `set_strategy.py`: A simplified implementation of the rewrite rule process, as described in Section 3.2 of `docs/implementation_v2/report.pdf`. The current implementation does not handle conditions on the properties of dummy variables, multiple dummy variables, or functions on the dummy variable. Instead, it offers basic rewrite rules for denesting, union, intersection, and set difference.
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

<!-- # TODO (non-implementation) -->

Name idea: Simile (or facsimile)

- kind of like simulation, also similar to modelling languages

# SYNT Presentation Plan

- Make a working PoC compiler
  - Handrolled frontend (scanner, parser)
  - Revise AST
  - Make a proper TRS to apply optimizations
  - Code gen for LLVM (either gen IR or use LLVM APIs)
  - Get warehouse and visitor info system examples working
- Make a Jupyter notebook for testing/explanations?
- Make a presentation, maybe a quick demo
-

# Running Simile in Jupyter

Run:

```powershell
pip install -ve packages/simile_compiler
pip install -ve packages/simile_jupyter_kernel
pip install -ve packages/simile_jupyter_extension
jupyter labextension develop --overwrite packages/simile_jupyter_extension
```

Check that the kernel is installed properly by running the following command without error:

```powershell
jupyter console --kernel simile_kernel
```

# Generating docs

```powershell
pydoctor --make-html --docformat=google --html-output=generated/api_docs/ packages/simile_compiler/
```
