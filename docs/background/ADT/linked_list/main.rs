struct LinkedList<T> {
    value: T,
    next: Option<Box<LinkedList<T>>>,
}

impl<T> LinkedList<T> {
    pub fn append(&mut self, value: T) {
        let new_node = Box::new(LinkedList { value, next: None });

        let mut l = self;
        while let Some(ref mut next_node) = l.next {
            l = next_node
        }
        l.next = Some(new_node);
    }

    pub fn concat(&mut self, other: LinkedList<T>) {
        let mut l = self;
        while let Some(ref mut next_node) = l.next {
            l = next_node
        }
        l.next = Some(Box::new(other));
    }
}
