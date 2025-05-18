
class Node<T(==)> {
  var value: T
  var next: Node?<T>

  ghost var repr: set<object>
  ghost var contents: seq<T>

  constructor (new_value: T)
    ensures Valid() && contents == [new_value] && repr == {this}
  {
    value, next := new_value, null;
    repr, contents := {this}, [new_value];
  }

  ghost predicate Valid()
    reads this, repr
  {
    && this in repr && |contents| > 0 && contents[0] == value
    && (next != null ==>
          && next in repr && next.repr <= repr && this !in next.repr
          && next.Valid() && next.contents == contents[1..])
    && (next == null ==> |contents| == 1)
  }

  method append(self: Node?<T>, new_value: T) returns (n: Node<T>)
    requires self == null || self.Valid()
    ensures n.Valid()
    ensures if self == null then n.contents == [new_value]
            else |n.contents| == |self.contents + [new_value]|
  {
    var new_node := new Node(new_value);
    new_node.value := new_value;
    new_node.next := null;

    if (self == null) {
      n := new_node;
      return;
    }
    n := self;

    var iter_list: Node<T> := self;
    var i := 0;
    while (iter_list.next != null)
      invariant
        iter_list.Valid() &&
        i + |iter_list.contents| == |n.contents| &&
        n.contents[i..] == iter_list.contents
      decreases |n.contents| - i
    {
      iter_list := iter_list.next;
      i := i + 1;
    }

    iter_list.next := new_node;
    // Update the ghost var repr for all nodes in the list...
  }

  method concat(self: Node?<T>, other: Node?<T>) returns (n: Node?<T>)
    requires self == null || self.Valid()
    requires other == null || other.Valid()
    ensures n != null ==> n.Valid()
    ensures n == null ==> (self == null && other == null)
    ensures self != null ==> (n != null && |n.contents| >= |self.contents|)
    ensures other != null ==> (n != null && |n.contents| >= |other.contents|)
  {
    if (self == null) {
      n := other;
      return;
    }
    n := self;

    var iter_list: Node<T> := self;
    var i := 0;
    while (iter_list.next != null)
      invariant
        iter_list.Valid() &&
        i + |iter_list.contents| == |n.contents| &&
        n.contents[i..] == iter_list.contents
      decreases |n.contents| - i
    {
      iter_list := iter_list.next;
      i := i + 1;
    }

    iter_list.next := other;
    // Update the ghost var repr for all nodes in the list...
  }
}
