# The Simile Compiler

A high level, optimizing model simulation language founded on set theory!

The repository is organized as follows:
- `.archive`: Older work that may still be relevant to future progress
  - `implementation`: Code for the first version of Simile, corresponding to the paper in `docs/implementation_v2`.
    <!-- - `rewrite_strategy_select_examples`: Collection of files used in `rewrite_strategy_select_examples_test.py`
      - `set_strategy.py`: A simplified implementation of the rewrite rule process, as described in Section 3.2 of `docs/ implementation_v2/report.pdf`. The current implementation does not handle conditions on the properties of dummy variables,   multiple dummy variables, or functions on the dummy variable. Instead, it offers basic rewrite rules for denesting, union,  intersection, and set difference.
      - `test_items.py`: Ground truth objects to test the rewrite rule set against. Generally, the process goes from plaintext ->   Simple AST -> Verbose AST (where sets are all in constructor form) -> Verbose AST with no set operations (only boolean  operations) -> Concrete AST -> Code output.
    - `ast_.py`: Collection of (empty-for-now) classes to represent the AST of our very high level language
    - `grammar.lark`: A CFG for our very high level language, written in a dialect suitable for the Lark Python library
    - `parser.py`: Contains a function that accepts a plaintext language input and returns an AST if valid in the language.
    - `README.md`: Progress tracker for the language
    - `rewrite_strategy.py`: Plan for rewrite pass of the optimizing compiler. Currently not implemented.
    - `rewrite_strategy_select_examples_test.py`: Select examples of rewrite strategy implementations, independent from the rest  of the compiler. This test suite relies on `implementation/rewrite_strategy_select_examples`
    - `transformer.py`: An API into lark that converts Lark Tree tokens into our custom AST classes.
    - `transformer_unit_test.py`: Unit tests for transformer.py -->
- `docs`: Everything documentation-related: Reports, Assignments, Specifications, and Papers.
  - `ADT`: A Survey of Abstract Data Types (in several languages)
  - `implementation_v2`: Revised report for a prototype set-theory based compiler. Corresponding implementation is in `.archive/implementation`.
  - `synt_2025`: Abstract submitted to the [SYNT 2025](https://synt2025.github.io/program.html) workshop
  - `term_rewriting`: A Survey of Term Rewriting for Compiler Optimization
  - `trs_specification`: Complete TRS Specification for the Simile Compiler
- `meetings`: Meeting notes for discussions involving Simile
- `notes`: Research notes
  - `examples`: Gathered/derived model simulation examples that make use of set theory
- `packages`: Python packages for Simile
  - `simile_compiler`: Actual contents of the compiler, from lexer to code generation
  - `simile_jupyter_extention`: TODO, a jupyter extension to recognize Simile syntax
  - `simile_jupyter_kernel`: TODO, a jupyter kernel to execute Simile
- `proposals`: Proposals discussing the problems and justification of our work
- `synt_demo`: Mostly a playground for messing with examples used in the SYNT 2025 demo. This will be archived after August 2.

<!-- # TODO (non-implementation) -->
<!--
Name idea: Simile (or facsimile)

- kind of like simulation, also similar to modelling languages -->

<!-- # SYNT Presentation Plan (Old)

- Make a working PoC compiler
  - Handrolled frontend (scanner, parser)
  - Revise AST
  - Make a proper TRS to apply optimizations
  - Code gen for LLVM (either gen IR or use LLVM APIs)
  - Get warehouse and visitor info system examples working
- Make a Jupyter notebook for testing/explanations?
- Make a presentation, maybe a quick demo -->

# Running Simile in Jupyter

**DOES NOT WORK YET**

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

# Generating API Documentation

```powershell
pip install pydoctor
pydoctor --make-html --docformat=google --html-output=generated/api_docs/ packages/simile_compiler/
```
