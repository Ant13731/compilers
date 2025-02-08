interface LinkedList2<T> {
	value: T;
	next: LinkedList2<T> | null;
}

function append2<T>(self: LinkedList2<T>, new_value: T): LinkedList2<T> {
	let l = self;
	while (l.next !== null) {
		l = l.next;
	}
	// Structural typing - No need to refer to LinkedList2 name explicitly
	l.next = { value: new_value, next: null };
	return self;
}

function concat2<T>(
	self: LinkedList2<T>,
	other: LinkedList2<T>,
): LinkedList2<T> {
	let l = self;
	while (l.next !== null) {
		l = l.next;
	}
	l.next = other;
	return self;
}
