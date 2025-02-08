type LinkedList<T> =
	| { kind: "nil" }
	| { kind: "cons"; value: T; next: LinkedList<T> };

function append<T>(self: LinkedList<T>, new_value: T): LinkedList<T> {
	if (self.kind === "nil") {
		return { kind: "cons", value: new_value, next: { kind: "nil" } };
	}

	let l = self;
	while (l.kind === "cons" && l.next.kind !== "nil") {
		l = l.next;
	}
	l.next = { kind: "cons", value: new_value, next: { kind: "nil" } };
	return self;
}

function concat<T>(self: LinkedList<T>, other: LinkedList<T>): LinkedList<T> {
	if (self.kind === "nil") {
		return other;
	}

	let l = self;
	while (l.kind === "cons" && l.next.kind !== "nil") {
		l = l.next;
	}
	l.next = other;
	return self;
}
