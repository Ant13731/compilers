import parser
import ast_
import random
import time

# from typing import TypeVar, Generic, Type, Any


birthday_book_src = """
# Note that the birthday book guarantees unique names (keys)
birthdays: dict[(str, int)] = { "Alice": 25, "Bob": 30, "Charlie": 35, "Charlie1": 35}
test_date: int = 25
"""
birthday_book_test_1 = (
    birthday_book_src
    + """
birthday_names: list[str] = (birthdays ^ -1)[{test_date}]
"""
    # """
    # for name in (birthdays ^ -1)[{test_date}]:
    #     print(name)
    # """
)

birthday_book_test_2 = (
    birthday_book_src
    + """
birthday_names: list[str] = []
for names, date in birthdays:
    if date == test_date:
        birthday_names: list[str] = names
        break
"""
    # """
    # for names, date in birthdays:
    #     if date == test_date:
    #         for name in names:
    #             print(name)
    # """
)


# res_src = parser.parse(birthday_book_src)
# res_test_1 = parser.parse(birthday_book_test_1)
# res_test_2 = parser.parse(birthday_book_test_2)


# Python cannot reverse dictionaries since keys must be hashable but values may not be


class BiDirectionalMapping[K, V]:
    def __init__(self, mapping: dict[K, V]):
        self.mapping = mapping
        self.reverse_mapping = {v: k for k, v in mapping.items()}

    def insert(self, key: K, value: V) -> None:
        self.mapping[key] = value
        self.reverse_mapping[value] = key

    def lookup(self, key: K) -> V:
        return self.mapping[key]

    def lookup_reverse(self, value: V) -> K:
        return self.reverse_mapping[value]


class BiDirectionalMappingManyToOne[K, V]:
    def __init__(self, mapping: dict[K, V]):
        self.mapping = {}
        self.reverse_mapping = {}

        for k, v in mapping.items():
            self.insert(k, v)

        # print(f"mapping: {self.mapping}")
        # print(f"reverse_mapping: {self.reverse_mapping}")

    def insert(self, key: K, value: V) -> None:
        if value not in self.reverse_mapping:
            self.reverse_mapping[value] = []
        self.mapping[key] = value
        self.reverse_mapping[value].append(key)

    def lookup(self, key: K) -> V:
        return self.mapping[key]

    def lookup_reverse(self, value: V) -> list[K]:
        return self.reverse_mapping.get(value, [])


class PartialRelationsByHashing[K, V]:
    def hash2(self, x: int) -> int:
        return x + 1

    def __init__(self, mapping: dict[K, V]):
        self.values: dict[int, V] = {}
        self.keys: dict[V, list[K]] = {}

        for k, v in mapping.items():
            self.insert(k, v)

    def insert(self, key: K, value: V) -> None:
        hk = hash(key)

        # hv = hash(value)
        # while hv in self.keys:
        #     hv = self.hash2(hv)
        # Hash collisions between values should be fine since we are just trying with many-to-one
        while hk in self.keys:
            hk += self.hash2(hk)

        self.keys[hk] = value
        if value not in self.values:
            self.values[value] = []

        self.values[value].append(key)

    def lookup(self, name: K) -> V | None:
        hashed_name = hash(name)

        # If two names happen to have the same hash, need to resolve the collision manually
        # This case is extremely rare though, so we expect only 1 execution iteration for this loop
        while True:
            possible_date = self.keys.get(hashed_name)
            # No entry found for this name
            if possible_date is None:
                return None

            names_corresponding_to_date = self.lookup_reverse(possible_date)
            if name in names_corresponding_to_date:
                return possible_date

            hashed_name = self.hash2(hashed_name)

    def lookup_reverse(self, date: V) -> list[K]:
        # Let python take care of any hash collisions here
        return self.values.get(date, [])


class RelationsByHashing[K, V]:
    def hash2(self, x: int) -> int:
        return x + 1

    def __init__(self, mapping: dict[K, V]):
        self.keys: dict[int, K] = {}
        self.vals: dict[int, V] = {}

        for k, v in mapping.items():
            self.insert(k, v)

        # print(f"keys: {self.keys}")
        # print(f"vals: {self.vals}")

    def insert(self, key: K, value: V) -> None:
        hk = hash(key)

        hv = hash(value)
        while hv in self.keys:
            hv = self.hash2(hv)
        # Hash collisions between values should be fine since we are just trying with many-to-one
        # while hv in self.vals:
        #     hv += self.hash2(hv)

        self.vals[hk] = value
        self.keys[hv] = key

    # TODO ONE OF THESE IS WRONG
    def lookup(self, key: K) -> V | None:
        # hk = hash(key)
        # while True:
        #     val = self.vals.get(hk)
        #     hv = hash(val)
        #     reverse_val = self.keys.get(hv)

        #     if reverse_val == key:  # should use is?
        #         return val
        #     hk = self.hash2(hk)
        hk = hash(key)
        value = self.vals.get(hk)
        hv = hash(value)

        if value is None:
            return None

        while True:
            # print(f"key: {key}, value: {value}, hv: {hv}")
            # check if the key is the same as the one we are looking for
            reverse_val = self.keys.get(hv)
            if reverse_val == key:
                return value
            hv = self.hash2(hv)
            reverse_val = self.keys.get(hv)
        # reverse_val = self.keys.get(hash(value))
        # if reverse_val == key:
        #     return value
        # else:
        #     # need to resolve collision
        #     raise KeyError(f"Key {key} not found in mapping")

    def lookup_reverse(self, value: V) -> list[K]:
        hv = hash(value)
        ret_list = []
        while True:
            key = self.keys.get(hv)
            hk = hash(key)
            reverse_key = self.vals.get(hk)

            if key is None:
                break

            if reverse_key == value:  # should use is?
                ret_list.append(key)
            hv = self.hash2(hv)
        return ret_list

        #  hashed_date = hash(date)

        # names: list[Name] = []
        # while True:
        #   possible_name = birthday_book_inverse.get(hashed_date)
        # #   No entry found for this name
        #   if possible_name is None:
        #     break

        #   date_corresponding_to_name = birthday_book_lookup(possible_name)
        #   if date_corresponding_to_name == date:
        #     names.append(name)

        # #   Need to check through all possible collisions for names corresponding to the input date
        #   hashed_date = collision_resolver(hashed_date)

        # return names


# test_dict = {
#     "Alice": 25,
#     "Bob": 30,
#     "Charlie": 35,
#     "Charlie1": 35,
# }
# rel_hash = PartialRelationsByHashing(test_dict)
# print("RelationsByHashing:")
# print("Lookups:")
# print("Lookup key: Alice", rel_hash.lookup("Alice"))
# print("Lookup key: Bob", rel_hash.lookup("Bob"))
# print("Lookup key: Charlie", rel_hash.lookup("Charlie"))
# print("Lookup key: Charlie1", rel_hash.lookup("Charlie1"))
# print()
# print("Reverse Lookups:")
# print("Lookup val: 25", rel_hash.lookup_reverse(25))
# print("Lookup val: 30", rel_hash.lookup_reverse(30))
# print("Lookup val: 35", rel_hash.lookup_reverse(35))
# print()
# print()
# bi_map = BiDirectionalMappingManyToOne(test_dict)
# print("BiDirectionalMapping:")
# print("Lookups:")
# print("Lookup key: Alice", bi_map.lookup("Alice"))
# print("Lookup key: Bob", bi_map.lookup("Bob"))
# print("Lookup key: Charlie", bi_map.lookup("Charlie"))
# print("Lookup key: Charlie1", bi_map.lookup("Charlie1"))
# print()
# print("Reverse Lookups:")
# print("Lookup val: 25", bi_map.lookup_reverse(25))
# print("Lookup val: 30", bi_map.lookup_reverse(30))
# print("Lookup val: 35", bi_map.lookup_reverse(35))

# More thorough testing
# l = 1000000
# r = 10000
# # rand_list_keys = list(map(str, random.choices(range(1, r), k=l)))
# # Assume keys are unique
# rand_list_keys = list(map(str, range(1, l)))
# rand_list_vals = list(map(str, random.choices(range(1, r), k=l)))
# rand_set_vals = list(set(rand_list_vals))
# print("first 10 keys: ", rand_list_keys[:10])
# print("first 10 vals: ", rand_list_vals[:10])
# rand_dict_1 = dict(zip(rand_list_keys, rand_list_vals))
# rand_dict_2 = dict(zip(rand_list_keys, rand_list_vals))

# start_time_1 = time.perf_counter()
# bi_map = BiDirectionalMappingManyToOne(rand_dict_1)
# insert_time_1 = time.perf_counter()
# for k in rand_list_keys:
#     # print(f"Lookup key: {k}", bi_map.lookup(k))
#     bi_map.lookup(k)
# lookup_time_1 = time.perf_counter()
# for v in rand_set_vals:
#     # print(f"Lookup val: {v}", bi_map.lookup_reverse(v))
#     bi_map.lookup_reverse(v)
# end_time_1 = time.perf_counter()
# print(f"BiDirectionalMappingManyToOne insert took {insert_time_1 - start_time_1:.6f} seconds")
# print(f"BiDirectionalMappingManyToOne lookup took {lookup_time_1 - insert_time_1:.6f} seconds")
# print(f"BiDirectionalMappingManyToOne reverse lookup took {end_time_1 - lookup_time_1:.6f} seconds")
# print(f"BiDirectionalMappingManyToOne took {end_time_1 - start_time_1:.6f} seconds")

# start_time_2 = time.perf_counter()
# hash_map = PartialRelationsByHashing(rand_dict_2)
# insert_time_2 = time.perf_counter()
# for k in rand_list_keys:
#     # print(f"Lookup key: {k}", hash_map.lookup(k))
#     hash_map.lookup(k)
# lookup_time_2 = time.perf_counter()
# for v in rand_set_vals:
#     # print(f"Lookup val: {v}", hash_map.lookup_reverse(v))
#     hash_map.lookup_reverse(v)
# end_time_2 = time.perf_counter()
# print(f"PartialRelationsByHashing insert took {insert_time_2 - start_time_2:.6f} seconds")
# print(f"PartialRelationsByHashing lookup took {lookup_time_2 - insert_time_2:.6f} seconds")
# print(f"PartialRelationsByHashing reverse lookup took {end_time_2 - lookup_time_2:.6f} seconds")
# print(f"PartialRelationsByHashing took {end_time_2 - start_time_2:.6f} seconds")
# print()
