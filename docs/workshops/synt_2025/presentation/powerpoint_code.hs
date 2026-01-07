S = [(x, y) | x <- [1 .. 100], y <- [1 .. 100]]

T = []

S ++ T

-- Desugaring step
-- Example credit: https://stackoverflow.com/questions/8029046/removing-syntactic-sugar-list-comprehension-in-haskell
xs = [1 .. 4]

ys = [1 .. 4]

pairList = [(i, j) | i <- xs, j <- ys]

-- ~>
concatMap (\i -> map (\j -> (i, j)) ys) xs
