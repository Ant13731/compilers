The following examples were taken from the SuFu repository

# autolifter
## dac
### 0after1.f
Problem: Does a list contain a 0 that appears after at least 1

Nice form:
```
exists i . i in 0 .. len(xs) and xs[i] == 0 and 1 in xs[:i]
```

Optimized form:
```python
seen_one = False
for x in xs:
    if x == 0 and seen_one:
        return True
    if x == 1:
        seen_one = True0
return False
```

Derivation:
```c
exists i . i in 0 .. len(xs) and xs[i] == 0 and 1 in xs[i:]
~>
for i in 0 .. len(xs):
    if xs[i] == 0 and 1 in xs[i:]: # efficient only if we can reinterpret xs as a set (which is probably costly...)
        return True
return False
~>
for i in 0 .. len(xs):
    if xs[i] == 0:
        if 1 in xs[i:]:
            return True
return False
~>
for i in 0 .. len(xs):
    if xs[i] == 0: // Observation: when we enter this if statement, we don't really need to check future iterations of i (we already loop over the rest of the list inside the if statement)
        for x in xs[i:]:
            if x == 1:
                return True
return False
~>
for i in 0 .. len(xs):
    if xs[i] == 0:
        for x in xs[i:]:
            if x == 1:
                return True
        return False // this derivation is now linear instead of quadratic
return False
~>
if_statement_activated = False
for i in 0 .. len(xs):
    if xs[i] == 0:
        if_statement_activated = True
    if xs[i] == 1 and if_statement_activated:
        return True
return False
~>
if_statement_activated = False
for x in xs:
    if x == 0:
        if_statement_activated = True
    if x == 1 and if_statement_activated:
        return True
return False
```

Optimized form (bitvector):
```python
# if this condition is satisfied, there would be a 01 next to each other in that order
return not xs and xs << 1
```

### 0s1s.f
Problem: Do all 1s occur before all 0s in a list xs: list[0 | 1]

Nice form:
```
all i . i in 0 .. len(xs) and (xs[i] == 0) ==> 1 not in xs[i:]
== not (exists i . i in 0 .. len(xs) and x[i] == 0 and 1 in x[i:])
all i . i in 0 .. len(xs) and (xs[i] == 1) ==> 0 not in xs[:i]
```

Optimized form (iterative):
```python
seen_zero = False
for x in xs:
    # Only care about numbers once we see a zero
    if x == 0:
        seen_zero = True
    if x == 1 and seen_zero:
        return False
return True
```

Derivation:
```c
all i . i in 0 .. len(xs) and (xs[i] == 0) ==> 1 not in xs[i:]
~>
not exists i . i in 0 .. len(xs) and not ((xs[i] == 0) ==> 1 not in xs[i:])
~>
not exists i . i in 0 .. len(xs) and not not ((xs[i] == 0) and not 1 not in xs[i:])
~>
not exists i . i in 0 .. len(xs) and xs[i] == 0 and 1 in xs[i:]
~>
for i in 0 .. len(xs):
    if xs[i] == 0 and 1 in xs[i:]:
        return False
return True
~> ... follow steps above
```

Derivation:
This one would require a reverse iteration (len(xs) .. 0) if we wanted to follow the above derivation. Alternatively, we could try reversing the indices:
```
all i . i in 0 .. len(xs) and (xs[i] == 1) ==> 0 not in xs[:i]
~>
all i . i in 0 .. len(xs) and (xs[i] == 0) ==> 1 not in xs[i:] # prefer the xs[i:] form for forward iteration
```


Optimized form (bitvector):
```python
# if this condition is not satisfied, there would be a 01 next to each other in that order
does_one_occur_after_zero = not xs and xs << 1
return not does_one_occur_after_zero
```

### 2nd-min.f
Problem: Find the second minimum of a set (list) of elements

Nice form:
```
min(S - {min(S)})
```

Optimized form:
```python
min1 = inf
min2 = inf
for x in S:
    if x < min1:
        min2 = min1
        min1 = x
        continue
    if x < min2:
        min2 = x
```

Derivation:
```c
min(S - {min(S)})
~>
min x . x in S and x != min(S)
~>
min2 = inf
for x in S:
    if x < min2 and x != min(S):
        min2 = x
return min2
~>
min2 = inf
for x in S:
    if x < min2:
        if x != min(S)
            min2 = x
return min2
~>
min2 = inf
for x in S:
    if x < min2:
        min1 = inf
        for x_ in S:
            if x_ < min1:
                min1 = x_
        if x != min1:
            min2 = x
return min2
~>
min1 = inf
min2 = inf
for x in S:
    for x_ in S:
        if x_ < min1:
            min1 = x_
    if x < min2:
        if x != min1:
            min2 = x
return min2
~>
min1 = inf
min2 = inf
for x in S:
    if x < min1:
        min2 = min1 // x != min1 guaranteed since we reassign it later
        min1 = x
        continue
    if x < min2:
        min2 = x // x != min1 guaranteed since we have uniqueness and min1 would have been caught by the prior if statement
return min2
```

But how useful is this to generalization? We need to be able to write tuple transformations within our set comprehension. We should be asking how min(S - {min(S)}) relates to min(S)

Trying another derivation:
min(S - {min(S)})

Tupled form of minimum:
```c
min(S) = m where m = choice(S) and forall x . x in S | m <= x
~>
m1 = inf
for x in S:
    if x <= m1:
        m1 = x
return m1
```

```c
sndmin(S) = (m1, m2)[1] where m1 = choice(S) and m2 = choice(S) and forall x . x in S and m1 <= x and (x > m1 ==> m1 < m2 <= x)
~>
m1, m2 = inf, inf
for x in S:
    if x < m1:
        m1, m2 = x, m1 # but how can we know to compute snd min if the fst min changes
    else if m1 < x < m2:
        m1, m2 = m1, x
```
After reading the Tupling Calculation Eliminates Multiple Data Traversals paper, it seems like the function we really want is:
```c
fold(f, xs) where f =
    if x < m1: (x, m1)
    elif x < m2: (m1, x)
    else: (m1, m2)
```

With f, then we can do:
```c
min(S - min(S))
~>
fold(min, S - fold(min,S))
~> # How do we bridge this gap?
fold(f, S)

fold(min, S - fold(min,S))
~>
for x in S:
    m1 = best min known yet
    m2 = best min known yet that is greater than m1
```

It really seems like for 2nd min, 3rd min, etc, the general problem might be easier. So instead we can do:
```
min(n, S) = sort(S)[n]
```

The nth min is fundamentally recursive:
```
min(n,S) = min(S - union i in 0..n | min(i,S)
```
this would use a list as our accumulator - sort the nth minimum elems
Optimized form:
```python
accumulator = []
for x in S:
    for i in 0..acc:
        if x < acc[i]:
            acc[i], acc[i+1:n-1] := x, acc[i:n-2] # chop off the last elem when we push the new min
            exit inner loop
return acc[n-1]
```

- this is 0(nlen(S)), we sort the first n elems through insertion sort
- min(n,S) = sort(S)[n] # but we only have to sort up to n or |S|-n elems (can estimate direction to iterate based on traits). sort elements lazily as needed. TODO figure out how to write sort(S) as a set theory expr then apply n to get this rewrite. sort(S) will necessarily create a new list since a set cannot be ordered
- n < log(len(S)) otherwise sorting the entire list is probably better for runtime (but not for memory if input is a set since we would need to create a new obj)

### 3rd-min.f
Problem: Find the third minimum of a set (list) of elements

Nice form:
min(S - {min(S), sndmin(S)})

Optimized form:
```python
min1 = inf
min2 = inf
min3 = inf
for x in S:
    if x < min1:
        min3 = min2
        min2 = min1
        min1 = x
        continue
    if x < min2:
        min3 = min2
        min2 = x
        continue
    if x < min3:
        min3 = x
```

Extension: n-min?

### atoi.f
Problem: Convert an array of single-digit ints into a number (ex. [1,2,3] -> 123)

Nice form:
```
sum i . i in 0 .. len(xs) | xs[i] * (10 ** i)
```

Optimized form:
```python
ret = 0
for i in 0 .. len(xs):
    ret += xs[i] * 10 ** i
return ret
```

### average.f
Problem: Calculate the average of a set of numbers

Nice form:
```
sum(S) / card(S)
```

Derivation:
```c
(sum x . x in S) / (count x . x in S)
~>
(iter x in S : c := 0 | c += x) // iterator : initializer | updater
/
(iter x in S : c := 0 | c += 1)
~>
sum_, count_ := iter x in S : c_1, c_2 := 0,0 | c_1, c_2 += x, 1
sum_ / count_
~>
c_1 := 0
c_2 := 0
for x in S:
    c_1 += x
    c_2 += 1
sum_ = c_1
count_ = c_2
sum_ / count_
```

Optimized form:
```python
sum_ = 0
card_ = 0
for x in S:
    sum_ += x
    card += 1
return sum_ / card_
```

### balanced.f
Problem: Does the balance of numbers ever dip below zero when iterating in order?

Nice form:
```
all(map \x -> x > 0 (map sum (heads xs)))       // map sum heads is basically scan sum
==
forall i . i in 0 .. len(xs) and sum(xs[:i]) > 0
```

Derivation:
```c
all(map \x -> x > 0 (map sum (heads xs)))
~>
forall x in heads(S) | sum(x) > 0
~>
forall i in 0 .. len(S) | sum(S[:i]) > 0
~>
forall i in 0 .. len(S) | sum x in S[:i] > 0
~>
forall i in 0 .. len(S) | sum_fc x in S[:i] > 0
```

Derivation (attempt with iter init update form):
```c
all(map \x -> x > 0 (map sum (heads xs)))
~>
forall x in heads(S) | sum(x) > 0
~>
forall i in 0 .. len(S) | sum(S[:i]) > 0
~>
forall i in 0 .. len(S) | sum x in S[:i] > 0
~> // also mark r as the return value we care about (so we know to exit early whenever it becomes false)
iter i in 0 .. len(S) : r := true | (iter x in S[:i] : c := 0 | c += x) > 0
~>
iter i in 0 .. len(S) : r := true | (iter j in 0 .. i : c := 0 | c += S[i]) > 0
~> // j is a subset of i, and r is updated only after the latest c is calculated
iter i in 0 .. len(S): c,r := 0, true | c += S[i]; r := r and c > 0
~>
iter x in S: c,r := 0, true | c += x; r := r and c > 0
~>
c := 0
r := true
for x in S:
    c += x
    r := r and c > 0
    // How can we make the exit-early observation? We know r is the only return value we really care about
    // and if r is false, the rest of the iterations will always be false...

```


Optimized form:
```python
running_sum = 0
for x in xs:
    running_sum += x
    if running_sum < 0:
        return False
return True
```

### cnt1s.f
Does this benchmark have a bug? `f h cnt t` instead of `f h pre t`?
Problem: Count the number of times 0,1 appears in the list

Nice form:
```c
// count == count the number of true values there are, from gries science of programming
count(i . i in 1 .. len(xs) and (xs[i-1], xs[i]) == (0,1))
// This translates pretty directly
```

Haskell way would be:
```
length [() | (0,1) <- zip xs (tail xs)]
```

Optimized form:
```python
counter = 0
for i = 1; i < len(xs); i++:
    if xs[i-1] == 0 and xs[i] == 1:
        i++
        counter++
return counter
```

### cnt1s2s3s.f
Problem: Count the number of occurrences of 12*3 in the list
Similar to above, need flags/FSM to solve this...

### count10p.f
Likely another flag/FSM
### count10s2.f
Likely another flag/FSM


### dropwhile.f
Problem: Count the number of positive elements before the first negative element

Nice form:
```c
count i . i in 0 .. len(xs) and (forall x . x in xs[:i] and x > 0)
~>
count i . i in 0 .. len(xs) and (forall j in 0 .. i | xs[i] > 0)
~>
iter i in 0 .. len(xs) : c := 0 | if (iter j in 0 .. i : r := true | r := r and xs[i] > 0): c += 1
~>
iter i in 0 .. len(xs) : r, c := true, 0 | r := r and xs[i] > 0; if r: c += 1
~>
iter i in 0 .. len(xs) : c := 0 | if xs[i] < 0: return c else: c += 1
```

Optimized form:
```python
positive_nums_before_first_negative = 0
for x in xs:
    if x < 0:
        return positive_nums_before_first_negative
    positive_nums_before_first_negative += 1
return positive_nums_before_first_negative
```

### is_sorted.f
Problem: Check if a list is sorted (strictly increasing each element though)

Nice form:
```c
forall i . i in 1..len(xs) and xs[i-1] < xs[i]
// Need a rule that knows when iters should terminate early... could explore cases of possible options?
~>
iter i in 1..len(xs) : r := true | r := true and xs[i-1] < xs[i]
~>
iter i in 1..len(xs) : r := true | if xs[i-1] > xs[i]: r := False
~> // and if r is false, then iteration terminates
iter i in 1..len(xs) : r := true | if xs[i-1] > xs[i]: (r := false; return)
```

Optimized form:
```python
for i in 1..len(xs):
    if xs[i-1] >= xs[i]:
        return False
return True
```

How can we know to exit early...

### largest_peak.f
Problem: Find the maximum sum of contiguous positive numbers

Nice form:
```
max(
    map(
        sum,
        filter(
            is_sorted and is_positive,
            inits(tails(xs))    // get all possible subsegments
        )
    )
)
==
max(s . s in inits(tails(xs)) and is_sorted(x) and all_positive(x) | sum(s))
```

Optimized form:
```python
current_sum = 0
max_sum = 0
for x in xs:
    if x > 0:
        current_sum += x
    else:
        current_sum = 0
    max_sum = max(max_sum, current_sum)
return max_sum
```

More general form (without positive restriction)
Problem: Find the max sum of contiguous numbers

Nice form:
<!-- max(map sum inits(tails xs))
==
max s in inits(tails(xs)) | sum(s)
==
max s in segments(xs) | sum(s)
~>
iter s in segments(xs) : m := -inf | m := max(m, sum(s))
~>
iter s in inits(tails(xs)) : m := -inf | m := max(m, sum(s))
~>
iter i in 0..len(tails(xs)) : m := -inf | m := max(m, sum(tails(xs)[:i]))

// Somehow we need to make the observation that we dont really want to consider all heads of tails - only the best head from each tail
max i in 0..len(xs), j in 0..len(xs) | (sum k in i..j | xs[k])
~>

...
max i in 0..len(xs) | max(xs[i], (max j in 0..i | sum(xs[j:i])) + xs[i])
~>
max i in 0..len(xs) | max(xs[i], mss(xs[:i]) + xs[i]) // are we sure this is right?
~>
max x in xs : c := -inf | c := max(x, c + x); c
~>
iter x in xs : current, best := -inf, -inf | c := max(x, c + x); b = max(b, c)

max s in tails(heads(xs)) | sum(s)
~>
max i in len(xs)..0 | max s in heads(xs) | sum(s)
~> -->

```c
max i in 0..len(xs), j in 0..i | sum(xs[j:i])
~>
max i in 0..len(xs) | (max j in 0..i | sum(xs[j:i]))
~> // max prefix sum up to i, over all i. use mts-like derivation
max i in 0..len(xs) | (iter j in 0..i: r := -inf | r := max(xs[j],r + xs[j]))
~> // j is iterating over the same values as i, with potential repeated computations
// well theres no actual reason j needs to start at 0 every time - can just compute the j instance along the way as the outer max. it can be represented by one number
// we essentially want to save the best value of this j sum (on the fly) - best suffix over all prefixes
max i in 0..len(xs) : r,m := -inf, -inf | r := max(xs[j], r + xs[j]); m := max(m, r)
```

Optimized form:
```python
current = xs[0]
best = xs[0]
for x in xs[1:]:
    current = max(x, current + x)
    best = max(best, current)
return best
```

### length.f
Skipping
### line_sight
Basically a maximum computation
### lis
Problem: longest increasing subsequence
The file here deals with longest contiguous increasing subsequence, but the generalized form may be more interesting

Nice form:
// get the max increasing subsequence using a max with function
max_with_function(
    lambda t: sum(t), // could also be length depending on what we are looking for
    {s . s in subsequences(xs) and is_strict_sorted(s)}
)

### longest_odd10s
Problem: max len of segment if the segment length is odd

Nice form:
```
max(
    map(sum,
        filter(
            lambda seq: len(seq) is odd
            subsequences(xs)
        )
    )
)
```

// From chatgpt
```
max { sum(S) | S ⊆ xs, S is a subsequence, |S| is odd }
```

### Skipping from logest00s to mps_p
### mps/mss?
Problem: maximum subarray sum (similar to mts)
### msp
Problem: max of any subsequence product from any position in the list

Nice form:
```
max (map product (tails xs))
```

The other max subseq/prefix benchmarks seem to follow this style - Kadane's algorithm
## lsp?
Longest Subsequence Property search
- longest subsequence satisfying a predicate

Nice form:
```
maxWith(
    lambda t: number to judge t's longest/maximum,
    subsequences(xs)
)
```

## segment-tree
Reiterations of the previous problems but using a segment tree to divvy up info about the collection (basically a data structure/tree of subsections of the list that allow for faster querying on part of the list)
## single-pass
Separates the function from the traversal of the list. much more readable than the other benchmarks
### mas
Problem: max achievable subarray sum. has two separate counters (like relu but for addition and subtraction. when the current sum is negative, its reset to zero)


# fusion
## algprog
### ex3.13
Problem: tri-sum

Nice form:
```
// [a,b,c] becomes
// [(a,0) (a,1) (a,2) (b,1) (b,2) (c,2)]
tri xs = flatmap (\x,ts -> map \t -> (t, x) ts) (enumerate (tails xs))
sum (map product (tri xs))
==
tri xs = {i,d . i in 0..len(xs) and depth in 0..i | (xs[i], d)}
trisum xs = sum i,d . i in 0..len(xs) and depth in 0..i | xs[i] * d
```

Derivation:
```c
sum i,d . i in 0..len(xs) and depth in 0..i | xs[i] * d
~>
sum i,d . i in 0..len(xs) | (sum d in 0 .. i | xs[i] * d)
~> // d can be replaced with closed form series values since multiplication distributes addition
sum i,d . i in 0..len(xs) | xs[i] * (sum d in 0 .. i | d)
~>
sum i . i in 0..len(xs) | xs[i] * (i * (i + 1) / 2)
```

Optimized form:
```python
total = 0
i = 0
for x in xs:
    total += x * i * (i + 1) // 2
    i += 1
return total
```

### page58

```
tri_op f xs = {i,d . i in 0..len(xs) and depth in 0..i | f(xs[i], d)}
==
sum i,d . i in 0..len(xs) and depth in 0..i | f(xs[i], d)
```

<!-- ### page60
finding height of a tree structure

### page62
Same as atoi with validation -->

## deforestation
### page7-1
```
sum(map square (1..n))
```
<!-- ### page7-2
sum square but tree version -->

## identities
### page3
```
min (map max xs)
```

### page5
```
sum (map product (tails xs))
```

### page6
```
segments = flatmap inits (tails xs)
max(map sum (segments xs))
```

## shortcut
### page1
```
all (map p xs)
```
### page3
```
sum(a..b)
```

### page7
basically average

### page8
```
// placement is the solution set
queens n = {placement . placement in powerset(n >< n) and card(placement)==n and is_bijection(placement) and safe(placement) | placement}
safe placement = forall pos1, pos2 in placement >< placement | pos1 != pos2 ==> abs(pos1.row - pos2.row) != abs(pos1.col - pos2.col)
```

<!--
forall pos1, pos2 in placement | pos1 != pos2 ==> abs(pos1.row - pos2.row) != abs(pos1.col - pos2.col)


{placement . placement in powerset(n >< n) and card(placement)==n and is_bijection(placement) and safe(placement) | placement}
~> // cardinality == n, so placement must be a subset of {_ |-> _} of len n
{placement . placement in powerset_with_size(n, 0..n >< 0..n) and is_bijection(placement) and safe(placement) | placement}
~>
{placement . placement in powerset_with_size(n, 0..n >< 0..n) and is_bijection(placement) and safe(placement) | placement}

~> // now lets tackle safety pruning
{placement . placement in 0..n <- -> 0..n and (forall pos1, pos2 in placement | pos1 != pos2 ==> abs(pos1.row - pos2.row) != abs(pos1.col - pos2.col)) | placement}


...
~> // the optimal solution is built on the observation that we can start with a valid n-1 partial placement and attempt to extend it
nqueens(n, current_row=0, placement={}) =
    if current_row == n:
        return {placement}
    else:
        return union c in 0..n and safe(placement \/ {(current_row, c)}) | nqueens(n, current_row+1, placement \/ {(current_row, c)})
~>



{x | x in P(0..n) and card(x) == n}
~>
{x | x in powerset_with_size(n, 0..n)}
// eg. let n == 3: then powerset_with_size(3, 0..3 >< 0..3) ==
(0,0) (0,1) (0,2)
(0,0) (0,1) (1,0)
(0,0) (0,1) (1,1)
(0,0) (0,1) (1,2)
...
powerset_with_size = {x in P(S) and card(x) == n | x}
powerset_with_size(n, S) = {x in S and y in powerset_with_size(n-1, S - {x}) | {x} \/ y} \/ powerset_with_size(n-1, S - {x})
-->

Derivation:

Maybe we cant generate the backtracking solution directly, but we can surely improve upon generating all powersets of (n >< n). We know that the solution has to have the constraints on cardinality and bijection at a minimum, so, lets first define a form for bijection:
```c
0..n <--> 0..n
~>
bijection(x: int=n, Y: set=0..n, current_bijection: set[<-/->]={}) =
    if x == 0: // no xs left
        return {current_bijection}

    return union y in Y | bijection(x-1, Y - {y}, current_bijection \/ {(n,y)})

X <--> Y
~>
bijection(X, Y, current_bijection={}) =
    if X == {}:
        return {current_bijection}
    x = choose(X)
    return union y in Y | bijection(X - {x}, Y - {y}, current_bijection \/ {(x,y)})
==
bijection(X, Y, current_bijection={}) =
    if X == {}:
        return {current_bijection}

    x = choose(X)
    extended_current_bijections = {}
    for y in Y:
        result = result \/ bijection(X - {x}, Y - {y}, current_bijection \/ {(x,y)})
    return result
```

Then a possible derivation:
```c
{placement . placement in powerset(n >< n) and is_bijection(placement) and safe(placement) | placement}
~> // cardinality implied by bijection's totality. <--> generates bijections between two sets. although niche, we can always observe this relationship between powerset, cartesian product, and bijection
{placement . placement in 0..n <--> 0..n and safe(placement) | placement}
~> // we need to partially consider only valid solutions - use a recursive formulation of bijection to build partial solutions
bijection_with_constraints(X=0..n, Y=0..n, current_bijection={}) =
    if X == {}:
        return {current_bijection}
    x = choose(X)
    return union y in Y and safe(current_bijection \/ {(x,y)})| bijection(X - {x}, Y - {y}, current_bijection \/ {(x,y)})
- OR -
bijection_ordered_with_constraints(x=n, Y=0..n, current_bijection={}) =
    if x == 0:
        return {current_bijection}
    return union y in Y and safe(current_bijection \/ {(n,y)})| bijection(n-1, Y - {y}, current_bijection \/ {(n,y)})
~>
bijection_ordered_with_constraints(x=n, Y=0..n, current_bijection={}) =
    if x == 0:
        return {current_bijection}
    return union y in Y and (forall pos1, pos2 in current_bijection \/ {(n,y)} >< current_bijection \/ {(n,y)} | pos1 != pos2 ==> abs(pos1.row - pos2.row) != abs(pos1.col - pos2.col)) | bijection(m-1, Y - {y}, current_bijection \/ {(n,y)})
~> // but we really only need to check safety of the latest placement, since we can assume all previous placements are already safe...
bijection_ordered_with_constraints(x=n, Y=0..n, current_bijection={}) =
    if x == 0:
        return {current_bijection}
    return union y in Y and (forall (old_x, old_y) in current_bijection | abs(old_x - x) != abs(old_y - y)) | bijection(m-1, Y - {y}, current_bijection \/ {(n,y)})
~>
bijection_ordered_with_constraints(x=n, Y=0..n, current_bijection={}) =
    if x == 0:
        return {current_bijection}
    return union y in Y and (forall (old_x, old_y) in current_bijection | old_x + old_y != x + y or old_x - old_y != x - y) | bijection(m-1, Y - {y}, current_bijection \/ {(n,y)})
~> // instead of iterating over all the old values, we can store them in a set
bijection_ordered_with_constraints(x=n, Y=0..n, current_bijection={}, plus_diag={}, minus_diag={}) =
    if x == 0:
        return {current_bijection}
    return union y in Y and x+y not in plus_diag and x-y not in minus_diag | bijection(m-1, Y - {y}, current_bijection \/ {(n,y)}, plus_diag \/ {x+y}, minus_diag \/ {x-y})
~> // no need to actually hold a set for y. can instead hold a set for the inverse (visited_columns == visited_ys). Do we actually need this though?
bijection_ordered_with_constraints(x=n, visited_ys={}, current_bijection={}, plus_diag={}, minus_diag={}) =
    if x == 0:
        return {current_bijection}
    return union y in 0..n and y not in visited_ys and x+y not in plus_diag and x-y not in minus_diag | bijection(m-1, visited_ys \/ {y}, current_bijection \/ {(n,y)}, plus_diag \/ {x+y}, minus_diag \/ {x-y})
==
nqueens(n, current_row=0, placement={}, cols={}, diag1={}, diag2={}) =
    if current_row == n:
        return {placement}
    return union c in 0..n and c not in cols and current_row + c not in diag1 and current_row - c not in diag2 | nqueens(n, current_row+1, placement \/ {(current_row, c)}, cols \/ {c}, diag1 \/ {row+c}, diag2 \/ {row-c})
```

Optimized form (all solutions):
```python
def queens(row, n, placement, cols, diag1, diag2):
    if row == n:
        return [placement]
    solutions = []
    for c in range(n):
        if c not in cols and \
           row + c not in diag1 and \
           row - c not in diag2:
            solutions.extend(
                queens(
                    row + 1,
                    n,
                    placement | {(row,c)},
                    cols | {c},
                    diag1 | {row+c},
                    diag2 | {row-c}
                )
            )
    return solutions
```


Optimized form (but this only finds one soln of nqueens - we want to look for all solutions):
```python
def safe(cols, r, c):
    for r2, c2 in enumerate(cols):
        if c2 == c or abs(r2 - r) == abs(c2 - c):
            return False
    return True

def dfs(r, cols):
    if r == n:
        return [cols[:]]

    res = []
    for c in range(n):
        if safe(cols, r, c):
            cols.append(c)
            res.extend(dfs(r + 1, cols))
            cols.pop()
    return res

return dfs(0, [])
```

## tupling
### page1
Tree - skip
### page8
fib(n) = ...

<!-- # synduce
## combine -->
### mss_with_sum
```
max (map sum (segments xs))
```
<!-- ## compressed_list
## constraints -->
### alist/count_eq
Problem: count the number of occurrences of a number in a list if it is unique

Nice form (can specifically optimize this existence question to sets very well):
```
if is_unique xs then contains w xs else 0
contains w = w in xs
```

This problem may be specifically about exiting loops early (it stops recursing when we know that w in xs)

Extend to finding the index of an element and exiting early

### alist/most_frequent
Problem: Return the value with the longest run length (given an input list like list[run length, value]) (assuming values are unique)
### alist/sum
Problem: Return the sum of matching values assuming the value is unique (this should always short-circuit and return the target if target in xs or 0)

### sorted_and_indexed
Problem: Given a sorted list, find the number of values < target minus number of values > target

Most of the files in this section are about exiting early assuming some property on the underlying collection

## expressions
### max_subexpr_sum
Problem: find the max value of an expression subtree
```
max { evaluate(x) | x is a subtree of xs}
```

<!-- ## indexed_list
## list
## list_to_tree
## misc
## nested_list -->
### mtss
Problem: Find the maximum running sum achievable by accumulating segment sums from a nested list of lists
```
max{max sum (segments xs) | xs in xss}
```

### pyramid_intervals
Problem: Return true if max - min of a list within a list of lists is in order
```
forall i . i in 1..xss and max(xss[i-1]) - min(xss[i-1]) < max(xss[i]) - min(xss[i])
```

<!-- ## numbers
## ptree
## sorted_list
## tailopt
Optimize simple tail recursion
## terms
## tree
## treepaths
## unimodal_lists
-->
## zipper
Rest of above are mostly tree based, which is not really the intended use case of sets anyhow. Perhaps we should consider nested sets as trees in this case? Half of these tree optimizations require constraints on the repr too, like a BST

# Rosetta code
## Birthday problem
Problem: What is the probability that 2 people share a common birthday?
Alt: Do two people share a common birthday? Equivalent to asking if a total functional relation is a bijection

Nice form:
```
birthdays: People -> Date
possible_birthdays = card({p |-> d . p |-> d in 1..n -> 1..365 | p |-> d})
possible_birthdays_without_collisions = card(i,j . i in 0..card(bdays) and j in 0..card(bdays) and i != j ==> bdays[i] != bdays[j] | i |-> bdays[i])
birthday_problem(birthdays) = 1 - possible_birthdays_without_collisions / possible_birthdays
```

# Misc
## Graph problems
Common problems related to graphs:
- find a cycle in a graph
- topological sort
- find connected components
- BFS, DFS
- Is a graph bipartite?
- Shortest path
- Max flow of a weighted directed graph

Notation:
- just use relations for directed graphs (bidirectional hashmap): `{0 |-> 1, 1 |-> 0, 2 |-> 3}`
- undirected graphs (can always rewrite to only use one-direction hashmap): `{0 |-> 1, 2 |-> 3}`
- weighted directed graph could tag weight metadata onto the bihashmap (like `0 |-> (1, 5)`) or use a set of records

# Maximum Tail Sum
Nice form:
```
max(map sum (tails xs))
==
{i . i in 1..n | sum xs[i:n]}
```

Library:
```
tails(S) = { i . i in 0 .. n | i..n <| S}
sum(xs) = sum ran(xs) // xs are treated as a relation. Sum doesnt care about order
```

Derivation:
```c
max(map sum tails(xs))
~>
max x in tails(xs) | sum(x)
~>
max x in {i . i in len(xs)..0 | xs[i:]} | sum(x)
~>
max i in len(xs)..0 | sum(xs[i:])
~>
max i in len(xs)..0 | sum(j in i..len(xs) | xs[j])
~>
max i in len(xs)..0 | sum(j in len(xs)..i | xs[j])
~>
max i in len(xs)..0 | (iter j in len(xs)..i : s := 0 | s += xs[j])
~>
max i in len(xs)..0 | (iter j in len(xs)..i : s := 0 | s += xs[j])
~> // overlapping ranges in the same direction
iter i in len(xs)..0 : s := 0, m := -inf | s += xs[j]; m := max(m, s)

```

Optimized form:
```python
mts = 0
cts = 0
for i in n..1:
    cts += xs[i]
    mts = max(mts, cts)
return mts
```

# What did we learn from these benchmarks?
- tuple folding category of optimizations
- search space/constraint dynamic programming (n queens)
- we should try to share nested iterators whenever possible. How can we generalize these rules?