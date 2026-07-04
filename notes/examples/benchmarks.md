The following examples were taken from the SuFu repository

# autolifter
## dac
### 0after1.f
Problem: Does a list contain a 0 that appears after at least 1

Nice form:
exists i . i in 0 .. len(xs) and xs[i] == 0 and 1 in xs[:i]

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

Optimized form (bitvector):
```python
# if this condition is satisfied, there would be a 01 next to each other in that order
return not xs and xs << 1
```

### 0s1s.f
Problem: Do all 1s occur before all 0s in a list xs: list[0 | 1]

Nice form:
all i . i in 0 .. len(xs) and (xs[i] == 1) ==> 0 xs[i:]

Optimized form (iterative):
```python
seen_zero = False
for x in xs:
    if x == 1 and seen_zero:
        return False
    # Only care about numbers once we see a zero
    if x == 0:
        seen_zero = True
return True
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
min(S - {min(S)})

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
sum i . i in 0 .. len(xs) | xs[i] * (10 ** i)

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
sum(S) / card(S)

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
all(map \x -> x > 0 (map sum (heads xs)))       // map sum heads is basically scan sum
==
forall i . i in 0 .. len(xs) and sum(xs[i]) > 0

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
// count == count the number of true values there are, from gries science of programming
count(i . i in 1 .. len(xs) and (xs[i-1], xs[i]) == (0,1))

Haskell way would be: length [() | (0,1) <- zip xs (tail xs)]

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
count i . i in 0 .. len(xs) and (forall x . x in xs[:i] and x > 0)

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
forall i . i in 1..len(xs) and xs[i-1] < xs[i]

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
max(
    map(sum,
        filter(
            lambda seq: len(seq) is odd
            subsequences(xs)
        )
    )
)

// From chatgpt
max { sum(S) | S ⊆ xs, S is a subsequence, |S| is odd }

### Skipping from logest00s to mps_p
### mps/mss?
Problem: maximum subarray sum (similar to mts)
### msp
Problem: max of any subsequence product from any position in the list

Nice form:
max (map product (tails xs))

The other max subseq/prefix benchmarks seem to follow this style - Kadane's algorithm
## lsp?
Longest Subsequence Property search
- longest subsequence satisfying a predicate

Nice form:
maxWith(
    lambda t: number to judge t's longest/maximum,
    subsequences(xs)
)

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
// [a,b,c]
// [(a,0) (a,1) (a,2) (b,1) (b,2) (c,2)]
tri xs = flatmap (\x,ts -> map \t -> (t, x) ts) (enumerate (tails xs))
sum (map product (tri xs))
==
tri xs = {i,d . i in 0..len(xs) and depth in 0..i | (xs[i], d)}
trisum xs = sum i,d . i in 0..len(xs) and depth in 0..i | xs[i] * d

### page58
tri_op f xs = {i,d . i in 0..len(xs) and depth in 0..i | f(xs[i], d)}
==
sum i,d . i in 0..len(xs) and depth in 0..i | f(xs[i], d)

### page60
finding height of a tree structure

### page62
Same as atoi with validation

## deforestation
### page7-1
sum(map square (1..n))
### page7-2
sum square but tree version

## identities
### page3
min (map max xs)

### page5
sum (map product (tails xs))

### page6
segments = flatmap inits (tails xs)
max(map sum (segments xs))

## shortcut
### page1
all (map p xs)
### page3
sum(a..b)

### page7
basically average

### page8
// placement is the solution set
queens n = {placement . placement in powerset(n >< n) and card(placement)==n and is_bijection(placement) and safe(placement) | placement}
safe placement = forall pos1, pos2 in placement | pos1 != pos2 ==> abs(pos1.row - pos2.row) != abs(pos1.col - pos2.col)

Optimized form:
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

# synduce
## combine
### mss_with_sum
max (map sum (segments xs))
## compressed_list
## constraints
### alist/count_eq
Problem: count the number of occurrences of a number in a list if it is unique

Nice form (can specifically optimize this existence question to sets very well):
if is_unique xs then contains w xs else 0
contains w = w in xs

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
max { evaluate(x) | x is a subtree of xs}

## indexed_list
## list
## list_to_tree
## misc
## nested_list
### mtss
Problem: Find the maximum running sum achievable by accumulating segment sums from a nested list of lists
max{max sum (segments xs) | xs in xss}

### pyramid_intervals
Problem: Return true if max - min of a list within a list of lists is in order
forall i . i in 1..xss and max(xss[i-1]) - min(xss[i-1]) < max(xss[i]) - min(xss[i])

## numbers
## ptree
## sorted_list
## tailopt
Optimize simple tail recursion
## terms
## tree
## treepaths
## unimodal_lists
## zipper

Rest of above are mostly tree based, which is not really the intended use case of sets anyhow. Perhaps we should consider nested sets as trees in this case? Half of these tree optimizations require constraints on the repr too, like a BST

# Rosetta code
## Birthday problem
Problem: What is the probability that 2 people share a common birthday?
Alt: Do two people share a common birthday? Equivalent to asking if a total functional relation is a bijection

Nice form:
birthdays: People -> Date
possible_birthdays = card({p |-> d . p |-> d in 1..n -> 1..365 | p |-> d})
possible_birthdays_without_collisions = card(i,j . i in 0..card(bdays) and j in 0..card(bdays) and i != j ==> bdays[i] != bdays[j] | i |-> bdays[i])
birthday_problem(birthdays) = 1 - possible_birthdays_without_collisions / possible_birthdays

