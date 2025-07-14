S, T = set(), set()
S = S | T  # Union
S = S & T  # Intersection
S = {x for x in S if x not in T}  # Difference

S, T = [], []
S = S + T  # Concatenation
S = [x for x in S if x not in T]


S, T = {}, {}
S.update(T)  # Relation Overriding
S = {x: y for x, y in S.items()}
S = {x: z for x, y in S.items() for y_, z in T.items() if y == y_}

U, V = set(), set()

(S | U) & V
# This expression is equivalent to:
ret = set()
for x in S:
    ret.add(x)
for x in U:
    ret.add(x)
for x in ret:
    if x not in V:
        ret.remove(x)
# with result ret

(S | U) & V -> (S & V) | ((U - S) & V)
# Transformed expression:
ret = set()
for x in S:
    if x in V:
        ret.add(x)
for x in U:
    if x not in S and x in V:
        ret.add(x)
# with result ret


