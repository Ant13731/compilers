Setting: search through a set of webpages with filters and sorts

Goal: return a list of results

```python

items: set[Item] = set()
keywords: set
order in {asc, desc}
sortby in {name, rating, ...}


def query_results(item: set[Item]) -> list[Item]:
    filtered_items = set(
        filter(
            lambda item: keyword in item,
            items
            )
        )
    return sort(sortby=sortby, order=order, filtered_items)
```
