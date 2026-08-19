Setting: Conway's game of life. A grid of pixels following a set amount of rules. Pixels may be white or black.

Goal: Determine image after X iterations

```python

# maybe get len and width information for the grid too
length = 100
width = 100

Point = tuple[int, int]
# initial configuration
Cells = set[Point]

def seed():
    return set()

def iteration(living: Cells) -> Cells:
    return living_cells_from_dead_cells(does_living_cell_live(living))

def neighbours(cell: Point, living: Cells) -> set[Point]:
    return set(
        filter(
            lambda other: abs(other.x - cell.x) <= 1 and abs(other.y - cell.y) <= 1,
            living
            ),
        )


def does_living_cell_live(cell: Point, living: Cells) -> bool:
    num_neighbours = len(neighbours(cell, living))
    if num_neighbours < 2 or num_neighbours > 3:
        return False
    return True


def living_cells_from_dead_cells(living: Cells) -> Cells:
    dead_cells = compliment(living)
    return living + set(
        filter(
            lambda dead_cell: neighbours(dead_cell, living) == 3,dead_cells
            )
        )

```
