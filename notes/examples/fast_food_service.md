```python

customer_line: sequence[Person]
items: set[Items]


menu: Item -> Price
orders: Person -/-> bag[Items]


```

Queries:
- How much will my order cost?
- How much money will we make?

This is nearly a duplicate of the warehouse system