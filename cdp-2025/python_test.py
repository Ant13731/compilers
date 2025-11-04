# Class credit: https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
class Bimap[K, V](dict[K, list[V]]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inverse: dict[V, list[K]] = {}

        for key, value in self.items():
            for val in value if isinstance(value, list) else [value]:
                self.inverse.setdefault(val, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            for v in self[key]:
                self.inverse[v].remove(key)
        super().__setitem__(key, value)
        for val in value if isinstance(value, list) else [value]:
            self.inverse.setdefault(val, []).append(key)

    def __delitem__(self, key):
        for v in self[key]:
            self.inverse.setdefault(v, []).remove(key)

        for v in self[key]:
            if v in self.inverse and not self.inverse[v]:
                del self.inverse[v]

        super().__delitem__(key)


def compose[K, V, T](self: Bimap[K, V], other: Bimap[V, T]) -> Bimap[K, T]:
    result = Bimap()
    for key in self:
        for v in self[key]:
            if v in other:
                result[key] = other[v]
    return result


def main():
    location = Bimap({"SYNT": 100, "ABC": 200, "CDP": 300})
    attends = Bimap({"Alice": "SYNT", "Bob": "ABC", "Charlie": "SYNT"})

    room = 100

    print(location.inverse)
    print(attends.inverse)

    print(compose(location.inverse, attends.inverse))

    num_meals = len((compose(location.inverse, attends.inverse))[room])
    print(num_meals)


main()
