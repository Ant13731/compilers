# Practical Alloy (documentation/book)

# Structural Modelling

- Signature declaration seems like classes with inheritance
- Analysis commands can find design problems or spec errors as early as possible
  - `run` for satisfiability checker (finds satisfying instance (witness?))
  - `check` for validity (finds counterexamples)
- Instances from analysis show up as graph like structures
  - atoms/nodes have no intrinsic semantics
- Object types can be disjoint or subset of a union of signatures, cardinality 1 == constant (singleton)
- Everything in Alloy is a relation
  - Simplifies syntax and semantics of the language
- Constraints on relations are added through facts
- Navigational style using `.` for relational joins/lookups removes the need for pointwise operations

## Example - File System

- relation for 1-to-many files per directory (type `dict[Dir, list[Entry]]`) looks like:
  ```Alloy
  sig Dir extends Object {
    entries: set Entry
  }
  ```
- then record types are just
  ```Alloy
  sig Entry extends Object {
    name: one Name
    obj: one Object
  }
  ```
