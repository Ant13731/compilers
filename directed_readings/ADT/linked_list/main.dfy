datatype LinkedList<T> = Nil | Cons(T, LinkedList<T>)

method append(lst: LinkedList<T>, x: T): LinkedList<T>
  ensures result == Cons(x, Nil) || result == Cons(lst.0, append(lst.1, x))
{
  match lst {
    case Nil => Cons(x, Nil)
    case Cons(y, ys) => Cons(y, append(ys, x))
  }
}

method concatenate(lst1: LinkedList<T>, lst2: LinkedList<T>): LinkedList<T>
  ensures result == lst1 || result == Cons(lst1.0, concatenate(lst1.1, lst2))
{
  match lst1 {
    case Nil => lst2
    case Cons(x, xs) => Cons(x, concatenate(xs, lst2))
  }
}
