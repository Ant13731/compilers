data LinkedList a = Nil | Cons a (LinkedList a)

append :: LinkedList a -> a -> LinkedList a
append Nil x = Cons x Nil
append (Cons y ys) x = Cons y (append ys x)

concatenate :: LinkedList a -> LinkedList a -> LinkedList a
concatenate Nil ys = ys
concatenate (Cons x xs) ys = Cons x (concatenate xs ys)

flatten :: LinkedList (LinkedList a) -> LinkedList a
flatten Nil = Nil
flatten (Cons xs xss) = concatenate xs (flatten xss)

instance Functor LinkedList where
    fmap f Nil = Nil
    fmap f (Cons x xs) = Cons (f x) (fmap f xs)

instance Applicative LinkedList where
    pure x = Cons x Nil
    Nil <*> _ = Nil
    _ <*> Nil = Nil
    (Cons f fs) <*> xs = fmap f xs `concatenate` (fs <*> xs)

instance Monad LinkedList where
    return x = Cons x Nil
    xs >>= f = flatten $ fmap f xs