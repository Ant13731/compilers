Setting: Ikea-like furniture warehouse with multiple parts, instructions, etc.
Or ex. Dofasco, need parts to upkeep machine, have internal store, multiple buyers

```python


material_catalogue: set[Material]
warehouse_inventory: bag[Material]

Ingredients = bag[Material]
product_catalogue = set[Ingredients]

# delivery rates?
# products that use the same parts?
# critical parts? lead times of parts?
    # ex. 2 products use the same part
# when do we need to order new parts? depends on delivery/consumption rates
# order vs storage vs in use
# numerical rates vs concrete information?

# See available inventory for sale
def get_available_products()

# Buy/sell 1 instance of a product
def add_product()
def remove_product()
def can_make_product()
```

Extensions:

- Capstone proj 8
  - calculate production of a chem plant model
- steel mill parts inventory
- general use inventory system
  - ex. people rent out parts, basically a library
