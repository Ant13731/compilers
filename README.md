# Compilers

Welcome to the very high level language repo! The repository is organized as follows:

```
└─ directed_readings: Reports and assignments for the COMPSCI 4Z03 Directed Readings course.
    └─ ADT: A Survey of Abstract Data Types (in several languages)
    └─ implementation: A report for the first version of a prototype compiler
    └─ implementation_v2: Revised report for a prototype compiler
    └─ term_rewriting: A Survey of Term Rewriting for Compiler Optimization
└─ implementation: Source code for prototypical compiler
    └─ ast_.py: Collection of (empty-for-now) classes to represent the AST of our very high level language
    └─ grammar.lark: A CFG for our very high level language, written in a dialect suitable for the Lark Python library
    └─ parser.py: Contains a function that accepts a plaintext language input and returns an AST if valid in the language.
    └─ transformer.py: An API into lark that converts Lark Tree tokens into our custom AST classes.
    └─ transformer_unit_test.py: Unit tests for transformer.py
└─ proposals: Proposals discussing the problems and justification of our work, WIP
```
