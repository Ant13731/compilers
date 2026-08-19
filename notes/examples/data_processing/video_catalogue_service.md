```python

content: set[Movie]
customers: set[Person]
subscription_plans: set[Plan]

subscribers: Person -/-> Plan
watch_history: Person <-> Movie

```

But this is another inventory system...