# 2.12 Partial expressions
- expressions are partial if they can be undefined
- when defined in context, it is well-formed
- if always defined, it is total
- In hoare triples, weakest preconditions must ensure expressions will be defined
    - eg. in x/y = 10, there is an implicit assert y != 0 before the statement
- formal meaning of short circuit operators:
    - DEFINED[[E && F]] = DEFINED[[E]] && (E ==> DEFINED[[F]])
    - DEFINED[[E || F]] = DEFINED[[E]] && (E || DEFINED[[F]])
    - DEFINED[[E ==> F]] = DEFINED[[E]] && (E ==> DEFINED[[F]])

# Termination and Recursion
- partial correctness if all terminating calls are correct (but no guarantee that a call terminates)
- use descending relations to prove termination