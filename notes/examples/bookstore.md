Setting: Online bookstore (facebook marketplace style, with sellers)

Goal: Check in/out books, keep track of stock, accounts, etc.

```python
struct Book:
    title, author, isbn: str
    price, rating: float
    description: str
    seller: Seller
    ... optional fields

struct Seller:
    name, phone, email: str
    ... optional fields

struct Customer:
    id: str
    auth: str
    ... optional fields

# States
stock: set[Book] # book holds the knowledge of its seller (since books have exactly 1 seller, this works out)
carts: mapping[Customer, Book] # is keeping track of all carts for every customer the right move? technically this is just extracting the relationship between customers and books from the struct. Ex. alternative struct for Customer would include cart: list[Book] param, but this might be too much like OOP?
featured_books: set[Book] # unique but ordered - list?
featured_sellers: set[Seller]

customers: set[Customer]
sellers: set[Seller]


# Invariant
assert range(carts) subseteq range(stock)

def register_customer(customer_name): ...
def register_seller(seller_name): ...

def put_book_for_sale(seller, book):
    stock |= {seller -> book}
def add_book_to_cart(customer, book):
    carts |= {customer -> book}
def cart_price(customer):
    return sum(map(lambda book: book.price, carts[customer]))
def buy_cart(customer):
    stock `setminus`= carts[customer]
    carts[customer] = {}

def add_all_books_from_author_to_cart(customer, author):
    #                                               get authors
    #                           match to author
    # add to cart
    carts[customer] |= filter(lambda a: a == author, map(lambda book:book.author, books))

def randomly_select_indices(length) -> set[int]: ...

def refresh_featured_books(stock):
    map(lambda (_, book): book, filter(lambda (i, book): i in randomly_select_indices(stock.len()) enum(stock)))
def refresh_featured_sellers(stock): ...
# Filters may include equality for keyword, author, comparison against const for price, rating, etc.
def query_for_book(stock, filters?, sort_by?):
    sort_by(filter(filters, stock))

# Concerns:
# - temporal logic - what happens if two customers order the same book at the same time?


```
