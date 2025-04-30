# Compiler for A Very High Level Language

Companion to the report in directed_readings/implementation_v2.

# Language Progress Tracker

- [ ] Compiler Frontend
  - [x] Create Lark grammar
  - [x] Create a basic AST representation for our language
  - [x] Create Lark Transformer to convert parsed grammar into AST form
  - [ ] Add in more context to Identifiers - eg. whether they are used as a definition or identifier
  - [ ] Remove superfluous Expr and Equivalence AST items (make them TypeAliases where needed)
    - [ ] Can also improve readability with more type aliases - one for Block = Statements | Expr, for example
    - [ ] Remake unit tests with updated AST
  - [ ] Add in better syntax error messages. This may need a custom parser separate from Lark, or to investigate how lark handles errors
  - [ ] Add an AST diff function for easier debugging (and settings to ignore superfluous Expr/EquivalenceStmt wrappers) - a functional equivalence function could be useful too, especially in making sure rewrite rules are correct
  - [ ] Make an AST -> plaintext generator for easier debugging and testing
- [ ] Compiler Passes
  - [ ] Make an AST traverser for simple passing
  - [ ] Pass 1: Create a lookup table for object Identifiers
    - [ ] Decide how to implement - involves scoping decisions, maybe attach a context/scope/environment to each node
  - [ ] Pass 2: Populate the AST with type information for each operation
    - [ ] Decide where to hold type information - likely within each AST node
    - [ ] Error handling for this
  - [ ] Pass 3: Optimization through rewrites
    - [ ] Sets
      - [ ] Implement a few rewrite rules for the report
      - [ ] Implement all stages
      - [ ] Stage three needs generator selection
    - [ ] Relations
    - [ ] Lists
    - [ ] Bags
    - [ ] Primitive types
    - [ ] Other...
  - [ ] Pass 4: Lower AST into LLVM IR
    - [ ] Make a system that can handle multiple targets (eg. WebAssembly)
