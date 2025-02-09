data LinkedList a = Nil | Cons a (LinkedList a)

append :: LinkedList a -> a -> LinkedList a
append Empty x = Cons x Empty
append (Cons y ys) x = Cons y (append ys x)

concatenate :: LinkedList a -> LinkedList a -> LinkedList a
concatenate Empty ys = ys
concatenate (Cons x xs) ys = Cons x (concatenate xs ys)