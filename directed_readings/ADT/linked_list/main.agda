data LinkedList (A : Set) : Set where
  Nil  : LinkedList A
  Cons : A → LinkedList A → LinkedList A

append : {A : Set} → LinkedList A → A → LinkedList A
append Nil x = Cons x Nil
append (Cons y ys) x = Cons y (append ys x)

concat : {A : Set} → LinkedList A → LinkedList A → LinkedList A
concat Nil l = l
concat (Cons x xs) l = Cons x (append xs l)

