datatype LinkedList<T> = Nil | Cons(T, LinkedList<T>)

function append<T>(lst: LinkedList<T>, x: T): LinkedList<T>
  decreases lst
{
  match lst {
    case Nil => Cons(x, Nil)
    case Cons(y, ys) => Cons(y, append(ys, x))
  }
}

function concatenate<T>(lst1: LinkedList<T>, lst2: LinkedList<T>): LinkedList<T>
  decreases lst1
{
  match lst1 {
    case Nil => lst2
    case Cons(x, xs) => Cons(x, concatenate(xs, lst2))
  }
}
