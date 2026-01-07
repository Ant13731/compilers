# The Simile Compiler

A high level, optimizing model simulation language founded on set theory!

The repository is organized as follows:
- `.archive`: Older work that may still be relevant to future progress
  - `implementation`: Code for the first version of Simile, corresponding to the paper in `docs/implementation_v2`.
- `docs`: Everything documentation-related: Reports, Assignments, Specifications, and Papers.
  - `background`: Technical reports on background information regarding programming language analysis and high-level optimization.
    - `ADT`: A Survey of Abstract Data Types (in several languages)
    - `implementation_v2`: Revised report for a prototype set-theory based compiler. Corresponding implementation is in `.archive/implementation` but has since been superseded by Simile.
    - `term_rewriting`: A Survey of Term Rewriting for Compiler Optimization
  - `spec`: Complete Language Specification for the Simile Compiler (a WIP on Overleaf; this version is updated rarely).
  - `thesis`: Masters thesis on Simile (WIP on Overleaf, rarely updated on here).
  - `workshops`: Abstracts and presentations about the underlying ideas behind Simile, delivered at various workshops.
    - `cdp_2025`: [CDP 2025](https://cdp-workshop.github.io/CDP/program/), co-located with [CASCON 2025](https://conf.researchr.org/home/cascon-2025)
    - `synt_2025`: [SYNT 2025](https://synt2025.github.io/program.html), co-located with [CAV 2025](https://conferences.i-cav.org/2025/)
- `meetings`: Meeting notes for discussions involving Simile
- `notes`: Research notes, important information is incorporated into `docs`
  - `examples`: Gathered/derived model simulation examples that make use of set theory
- `packages`: Python packages for Simile
  - `simile_compiler`: Actual contents of the compiler, from lexer to code generation
  - `simile_jupyter_extention`: TODO, a jupyter extension to recognize Simile syntax
  - `simile_jupyter_kernel`: TODO, a jupyter kernel to execute Simile
- `proposals`: Proposals discussing the problems and justification of our work (outdated)

<!-- Name idea: Simile (or facsimile) - kind of like simulation, also similar to modelling languages -->

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
