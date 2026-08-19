```python

record CellPoint:
    x: int
    y: int

record Point:
    x: float
    y: float

record CellEdge:
    point: CellPoint
    relative_edge: Down | Left # dont need up/right offsets (we just increase cellpoint as needed)
    is_solid: bool

record CellProperties:
    density: float
    temperature: float
    pressure: float


time_step: float
grid: set[CellPoint]
velocity_grid: CellEdge --> float # store velocities at the cell edges
property_grid: CellPoint --> CellProperties

divergence: CellPoint --> float
pressure: CellPoint --> float


procedure self_advect(velocity_grid: CellEdge --> float) -> CellEdge --> float:
    velocity_grid_as_cell_points = {c->v . c->v in velocity_grid | edge_to_point(c) -> v}
    return {c->v . c->v in velocity_grid and prev_pos = c.point - time_step * v| c -> bilinear_interpolate(prev_pos, velocity_grid_as_cell_points)}
procedure edge_to_point(edge: CellEdge) -> Point:
    if edge.relative_edge == Down:
        return edge.point + (0,0.5)
    return edge.point + (0.5,0)

procedure advect_property(velocity_grid: CellEdge --> float, property_grid: CellPoint --> CellProperties) -> : CellPoint --> CellProperties:
    velocity_grid_as_cell_points = {c->v . c->v in velocity_grid | edge_to_point(c) -> v}
    velocity_grid_offset_points = {c->v . c->v in velocity_grid | bilinear_interpolate(prev_pos, velocity_grid_as_cell_points) -> v}
    return {c->v . c->v in velocity_grid and prev_pos = c.point - time_step * v| c -> bilinear_interpolate(prev_pos, velocity_grid_offset_points)}


procedure divergence(velocity_grid: CellEdge --> float) -> CellPoint --> float:
    grid = {e->v . e->v in velocity_grid | e.point}
    return {c . c in grid | c -> divergence_at_cell(c, velocity_grid)}
procedure divergence_at_cell(c: CellPoint, velocity_grid) -> float:
    # get velocities from neighbours
    right = velocity_grid(CellEdge(c + (1,0), Left))
    left = velocity_grid(CellEdge(c, Left))
    top = velocity_grid(CellEdge(c + (0,1), Down))
    bottom = velocity_grid(CellEdge(c, Down))
    return right - left + top - bottom

procedure pressure_solve(pressure: CellPoint --> float, velocity_grid: CellEdge --> float) -> CellEdge --> float:
    new_pressure = {c . c in dom(pressure) | c -> pressure_solve_at_cell(c, pressure, velocity_grid)}
    return {e->v . e->v in velocity_grid | e -> pressure_resolved_velocity(e, v, new_pressure)}
# lagrangian operator nabla^2
procedure pressure_solve_at_cell(c: CellPoint, pressure: CellPoint --> float, velocity_grid: CellEdge --> float) -> float:
    right = pressure(c + (1,0))
    left = pressure(c - (1,0))
    top = pressure(c + (0,1))
    bottom = pressure(c - (0,1))

    p = right + left + top + bottom
    divergence = divergence_at_cell(c, velocity_grid)

    return p - divergence
procedure pressure_resolved_velocity(e: CellEdge, v: float, pressure: CellPoint --> float) -> float:
    if e.relative_edge == Left:
        return v - (pressure(e.point + (1,0)) - pressure(e.point))
    return v - (pressure(e.point + (0,1)) - pressure(e.point))



procedure bilinear_interpolate(pos: Point, property_grid: CellPoint --> Number) -> Number:
    # Find out what a property value is in between points on a grid
    x1 = floor(pos.x)
    x2 = ceil(pos.x)
    y1 = floor(pos.y)
    y2 = ceil(pos.y)

    q11 = property_grid(x1, y1)
    q12 = property_grid(x1, y2)
    q21 = property_grid(x2, y1)
    q22 = property_grid(x2, y2)

    property_across_x_at_y1 = ((x2-pos.x)/(x2-x1))*q11 + ((pos.x-x1)/(x2-x1))*q21
    property_across_x_at_y2 = ((x2-pos.x)/(x2-x1))*q12 + ((pos.x-x1)/(x2-x1))*q22
    return ((y2-pos.y)/(y2-y1))*property_across_x_at_y1 + ((pos.y-y1)/(y2-y1))*property_across_x_at_y2

```

Incompressible fluid simulation
- Modelled by a grid of cells containing info on velocity, density, temperature, etc. of the surrounding air
- Uses Navier-Stokes equations

Algorithm (for one point in time):
<!-- Only use basic version of the algo for now - pressure and advection go a long way
- Apply forces `f`
- Vorticity confinement -->
- Resolve pressure
- Self-advect
- Advect other properties
<!-- - Diffuse and decay -->