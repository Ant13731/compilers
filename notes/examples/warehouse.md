Setting: Ikea-like furniture warehouse with multiple parts, instructions, etc. This model can also be used in factories that need to upkeep machines

Goal/Interesting Questions:

- What products can we make given the available materials?
  - What about products that require the same materials? Becomes a resource problem similar to dynamic programming
- Given the typical weekly usage of materials, how much should we buy each week? How much should we store?

```python

material_catalogue: PartID -> FriendlyName
material_price_catalogue: PartID -> Price

# if elements in the bag are immutable, then storing a bag as a relation is likely to be much more efficient
inventory: PartID -> Quantity | bag[PartID] | set[tuple[PartID, Quantity]]


recipes: ProductID -> bag[PartID]
# or
# recipes: ProductID <-> tuple[PartID, Quantity]

def purchase_item(product_id: ProductID):
    used_materials = recipes[{product_id}]
    inventory = inventory - used_materials # bag minus
    return used_materials

# How much will a restock cost?
def restock(expected_inventory: bag[PartID]) -> Price:
    inventory_to_buy: bag[PartID] = expected_inventory - inventory # bag minus
    cost = sum(range(relational_composition(material_price_catalogue, inventory_to_buy))) # every entry in inventory_to_buy is mapped to a price, the cost is then summed
    return cost

# We know what items we sold, but say we don't have knowledge of past inventory
def weekly_profit(sold_items: bag[ProductID]) -> Price:
    sold_inventory = recipes[sold_items] # bag lookup - like set image but keep duplicates
    profit = sum(range(relational_composition(sold_inventory, material_price_catalogue))) # all materials are converted into prices
    return profit

def most_sold_material(sold_items: bag[ProductID]) -> PartID:
    sold_materials: bag[PartID] = relational_composition(recipes, sold_items)
    return max(organizeby=count, sold_materials) # if this was represented as a double hash table, we would just reverse lookup by the max number of occurrences in the bag.



```

Setting 2: A factory needs parts to upkeep a machine, has internal store with multiple competing buyers

<!-- ```python


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
  - ex. people rent out parts, basically a library -->
