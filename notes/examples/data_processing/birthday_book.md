# The Birthday Book Problem

A birthday book tracks the relationship between the names of people and their birthdays. All names are assumed to be unique, yet two or more people may share the same birthday. Using this book, we attempt to answer two main questions:

- When is the birthday of a particular person?
- Who has a birthday on a given date?

These questions correspond to the lookup and reverse lookup tasks on a set of relations. In other words, to find the birthday of a person, we require a function $lookup: Name \rightarrow Date$. Likewise, we denote $lookup^{-1}: Date \rightarrow \{Name\}$, which returns a set of names that are born on a specific date. Then, our high level language should provide an optimized version of these functions that returns from $lookup$ or $lookup^{-1}$ as fast as possible.

Extending the problem, we demonstrate the ergonomics of set operators over real world models. With consideration for the bidirectional mapping setup from the previous questions, we seek to efficiently implement the following scenarios:

- Two groups of friends become acquainted, each with an originally independent birthday book. Eventually, they wish to merge their records together, removing duplicate entries for mutual friends.
- A subset of the original friend group eventually have a falling out and wish to separate from the main group. This problem then requires one birthday book to be separated into two.
- When planning for a birthday party, the group wants to acknowledge not only the person's birthday, but any and all national holidays associated with that day.

In more formal notation:

```python
# State space
known: set[Name] = {}
birthday: mapping[Name, Date] = {}
holidays: mapping[Name, Date] = {}

# Invariant
known = dom(birthday)

# Functions
def add_birthday(name, date):
    if name not in known:
        birthday |= {name |-> date}
        known |= {name}

def find_birthday(name):
    if name in known:
        return birthday[{name}]

def remind_birthday(date):
    return birthday^-1[{date}]

def num_of_cards(date):
    return card(remind_birthday(date))

def join_friend_group(birthday_book_2: mapping[Name, Date]):
    birthday |= birthday_book_2

def remove_names(names: set[Name]):
    domain_subtraction(names, birthday) # Like event b operator

def remove_names_alt(names: set[Name]):
    domain_restriction(~names, birthday)

def find_holidays_on_persons_birthday(name):
    return (birthday ◦ holidays^-1)[name]
```
