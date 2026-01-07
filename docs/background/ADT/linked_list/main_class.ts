class LinkedList3<T> {
	value: T;
	next: LinkedList3<T> | null = null;

	append3(new_value: T) {
		let l: LinkedList3<T> = this;
		while (l.next !== null) {
			l = l.next;
		}
		l.next = new LinkedList3();
		l.next.value = new_value;
	}

	concat3(other: LinkedList3<T>) {
		let l: LinkedList3<T> = this;
		while (l.next !== null) {
			l = l.next;
		}
		l.next = other;
	}
}
