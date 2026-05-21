type resolver vs type checker

type resolver
- find types of every combination of expressions, skipping over the internal type operations
    - for example, the type of an assignment statement is none (since its not actually an expression)

type checker
- make sure all types make sense when put together
    - ex. an assignment statement should have left hand type == right hand type
    - ex. quantifier bound vars must match their generator types
- some type checking is interspersed through the symbol table and resolver functions...
- should just combine this then






# Type analysis
- reserved_keywords_check: AST -> bool
- symb_table = populate_symbol_table
    - populate_base_types
    - populate_builtin_functions
    - populate
        - scope
        - insert_symbols
        - ast_to_type
        - ast_to_trait
- AST = normalize_ast: AST
- assert_no_identifiers (parser_only nodes): AST -> bool
- synthesize_types: AST, symb_table -> type
- type_check: AST, symb_table -> bool (either pass/fail)











# Abstract Data Types to concrete data types
- what are the restrictions and requirements?

| Abstract Type | Refinement/Trait conditions | Concrete Type |
| ------------- | --------------------------- | ------------- |
| Set<T>        | None                        | Array<T>      |
|               | with hashable T             | HashSet<T>    |
|               | with orderable T            |

### Judgement list

<!-- Env $\emptyset$
Env $I$
Fetch Identifier
Reflexive Subtype
Transitive Subtype
Subsumption
Top Type
Sub Top Type
Sub Function
Sub Set
Sub Product
Sub Record
Type Refinement
Powerset
Emptyset
Emptyset Bottom
Set Enumeration
Primitives - bool
Primitives - int
Primitives - float
Primitives - str
Nat from int
Int from float
Tuples
Empty Tuple
Empty Tuple Bottom
Tuple Enumeration
Relation Type
Bag Type
Bag Enumeration
Sequence Type
Sequence Enumeration
Enum from static set
Relation Subtype - Total Relation
Relation Subtype - Surjective Relation
Relation Subtype - Total Surjective Relation
Relation Subtype - Partial Function
Relation Subtype - Total Function
Relation Subtype - Partial Injection
Relation Subtype - Total Injection
Relation Subtype - Partial Surjection
Relation Subtype - Total Surjection
Relation Subtype - Bijection
Variable Assignment
Type Alias Assignment
Type Alias
Refined Variable Assignment
Refined Type Alias
Command - break
Command - continue
Command - skip
Command - return
Lambda Expression
Quantification Body
Binds (with generator)
Binds with OR
Binds with AND
Structural Match
Structural Match with Tuple
General Union
General Intersection
Forall
Exists
Set Comprehension
Bag Comprehension
Sequence Comprehension
Binary Boolean Operations
Boolean Operations - Negation
Equals
Ordering Operators
Set Membership
Set Ordering Operations
Set Operations
Cartesian Product
Maplet
Numerical Range
Set Operations - Powerset
Bag Operations - (Max) Union
Bag Operations - Image
Relation Operations - Function Call
Relation Operations - Image
Relation Operations - Overriding
Relation Operations - Composition
Relation Operations - Domain Restriction
Relation Operations - Domain Subtraction
Relation Operations - Range Restriction
Relation Operations - Range Subtraction
Relational Subtype - Domain Restriction
Relational Subtype - Domain Subtraction
Relational Subtype - Range Restriction
Relational Subtype - Range Subtraction
Relational Subtype - Inverse
Relational Subtype - Overriding
Relational Subtype - Composition
Sequence Operations - Concatenation
Integer Operations - Division
Integer Operations - Modulo
Numerical Operations - Addition
Numerical Operations - Subtraction
Numerical Operations - Floating Division
Numerical Operations - Multiplication
Numerical Operations - Negation
Numerical Operations - Exponentiation
Records - Access
Records - Type Definition
Records - Initialization
Command - Composition
Command - Valid Import Module
Command - Valid Import Names
Command - Import Module
Command - Import Names
Command - Procedure Definition
Command - Procedure Call
Command - If
Command - If Else
Command - While
Command - For
Command - Block
Built-in - Minimum
Built-in - Mapped Minimum
Built-in - Maximum
Built-in - Mapped Maximum
Built-in - Choice
Built-in - Domain
Built-in - Range
Built-in - Cardinality
Built-in - Bag Size
Built-in - Sum
Built-in - Cast
Built-in - Cast With
Domain Empty
Literal implies a Domain
Literal within Domain
Orderable Domain without Min
Orderable Domain with Min
Orderable Domain without Max
Orderable Domain with Max
Min implies Order
Max implies Order
Full set
Empty Size
Non-Empty Size
Size implies Iterable
Orderable Literal is Min
Orderable Literal is Max
Less than with max-min
Less than or equal with max-min
Less than with min-max
Less than or equal with min-max
Upto
Is Empty
Set add
Set add with Total
Set delete
Set delete removes Total
Membership within domain
Membership within total domain
Enumeration
Cardinality
Powerset
Empty choice fails (statically)
Sum bounds
Product bounds
Min looks for Min
Max looks for Max
Union Literal
Union with widest Domain
Union with lowest Min
Union with highest Max
Union with Empty keeps Traits
Union with widest size
Union with Total stays
Generics with set ops
Cartesian Product
Total Cartesian Product
Subset with min-max
Inverse swaps Domain
Empty Composition
Image with Empty Arg
Image with Empty Relation
Image with ManyToOne
Image with ManyToOne
Total domain
Total range
Concat with Size -->

### Judgement scratch list

<!-- For the symbol table -->
Note: The symbol table is responsible for making sure we do not shadow identifiers!
Env $\emptyset$
Env $I$

<!-- For the types themselves -->
Subsumption
Top Type
Powerset
Emptyset

Emptyset Bottom

Primitives - bool
Primitives - int
Primitives - float
Primitives - str
Nat from int
Int from float
Tuples
Empty Tuple
Empty Tuple Bottom

Relation Type
Bag Type
Bag Enumeration
Sequence Type
Sequence Enumeration
Enum from static set

Variable Assignment
Type Alias Assignment
Type Alias
Refined Variable Assignment
Refined Type Alias

Lambda Expression - scope? symbols?
Quantification Body
Binds (with generator)
Binds with OR
Binds with AND
Structural Match
Structural Match with Tuple


Relational Subtype - Domain Restriction
Relational Subtype - Domain Subtraction
Relational Subtype - Range Restriction
Relational Subtype - Range Subtraction
Relational Subtype - Inverse
Relational Subtype - Overriding
Relational Subtype - Composition
Integer Operations - Division
Integer Operations - Modulo
Numerical Operations - Addition
Numerical Operations - Subtraction
Numerical Operations - Floating Division
Numerical Operations - Multiplication
Numerical Operations - Negation
Numerical Operations - Exponentiation

Records - Type Definition
Records - Initialization
Command - Composition
Command - Valid Import Module
Command - Valid Import Names
Command - Import Module
Command - Import Names
Command - Procedure Definition

Command - If
Command - If Else
Command - While
Command - For
Command - Block
Built-in - Minimum
Built-in - Mapped Minimum
Built-in - Maximum
Built-in - Mapped Maximum
Built-in - Choice
Built-in - Domain
Built-in - Range
Built-in - Cardinality
Built-in - Bag Size
Built-in - Sum
Built-in - Cast
Built-in - Cast With
Domain Empty
Literal implies a Domain
Literal within Domain
Orderable Domain without Min
Orderable Domain with Min
Orderable Domain without Max
Orderable Domain with Max
Min implies Order
Max implies Order
Full set
Empty Size
Non-Empty Size
Size implies Iterable
Orderable Literal is Min
Orderable Literal is Max
Less than with max-min
Less than or equal with max-min
Less than with min-max
Less than or equal with min-max
Upto
Is Empty
Set add
Set add with Total
Set delete
Set delete removes Total
Membership within domain
Membership within total domain
Enumeration
Cardinality
Powerset
Empty choice fails (statically)
Sum bounds
Product bounds
Min looks for Min
Max looks for Max
Union Literal
Union with widest Domain
Union with lowest Min
Union with highest Max
Union with Empty keeps Traits
Union with widest size
Union with Total stays
Generics with set ops
Cartesian Product
Total Cartesian Product
Subset with min-max
Inverse swaps Domain
Empty Composition
Image with Empty Arg
Image with Empty Relation
Image with ManyToOne
Image with ManyToOne
Total domain
Total range
Concat with Size

# ONPLS
15 minutes

Current state
1. Programming with Abstract Data Types
2. Ex. Visitor Information System - need to change the workshop names
3. Ex. Warehouse Inventory System
4. Abstract Data Types in programming + modelling languages
5. Goals
6. Related work
7. Ex. Optimizing data type operations in python
8. Concrete types within Visitor Info System
9. Naive Implementation of the VIS
10. Designing Optimizations
11. Term Rewriting for Hash-based Sets and Relations
12. Visitor Information System - Translation
13. Preliminary Results
14. Operation Running Time
15. Roadmap

Shortened version
16. Programming with Abstract Data Types
17. Ex. Visitor Information System - need to change the workshop names
19. Abstract Data Types in programming + modelling languages
20. Goals
21. Related work
22. Ex. Optimizing data type operations in python
23. Concrete types within Visitor Info System
24. Naive Implementation of the VIS
25. Designing Optimizations
26. Term Rewriting for Hash-based Sets and Relations
27. Visitor Information System - Translation
28. Preliminary Results
29. Operation Running Time
30. Roadmap
Extra slides
31. Ex. Warehouse Inventory System
