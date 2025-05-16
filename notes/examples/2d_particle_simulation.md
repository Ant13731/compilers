Setting: a finite number of (ideal gas) particles in a bounded box

Goal: simulate collisions between particles in 2D, update positions every 1 unit of time

- time is modelled by iterations

```python
type Vec: tuple[float, float] # x, y

struct Particle:
    p: Vec # position
    v: Vec # velocity
    # a: Vec # acceleration (unused? mostly for if there were charged particles)
    k: int # charge, either 1 or -1

const time_unit: float = 1 # increment time by 1 sec
const collide_distance: float = 1e-4
const boundary: Shape

particles: Multiset[Particle]

# No side effects
def iteration(particles: Multiset[Particle]) -> Multiset[Particle]:
    next_velocity(next_acceleration(next_positions(calculate_collisions(particles))))

### Initial collisions with other particles
def calculate_collisions(particles) -> Multiset[Particle]:
    return map(collide_with_walls, map(collide, {p -> particles | p in particles}))

def collide_with_walls(p):
    if p.p.x outside of boundary:
        p.v.x = - p.v.x
    if p.p.y outside of boundary:
        p.v.y = - p.v.y
    return p

# 1 particle collides with possibly all other particles
def collide(p, particles):
    return Particle(
        p: p.p
        v: vec_sum((p_.v - p.v) * dot(p.v - p_.v, p.p - p_.p)/(distance(p, p_)) | p_ in particles and distance(p, p_) < collide_distance)
        # a:
    )

### Moves particles 1 iteration forward
def next_positions(particles) -> Multiset[Particle]:
    return {leap_frog_first_half(p_) | p in particles}

# Move 1 particle forward as determined by innate velocity
def leap_frog_first_half(particle):
    next_half_velocity = particle.v + particle.a * time_unit/2
    next_position = position.x + next_half_velocity * time_unit
    return Particle(next_position, next_half_velocity, particle.a)

### Modifies acceleration based on other particles
def next_acceleration(particles):
    return map(new_accelerations, {p -> particles | p in particles})

def map_accelerations(p, particles):
    return Particle(
        p: p.p
        v: p.v
        a: vec_sum((p.p - p_.p).normalize() * p.k * p_.k / distance(p, p_) | p_ in particles)
    )

### Moves velocity based on acceleration
def map_velocity(particles):
    return {leap_frog_snd_half(p_) | p in particles}

# Calculate new velocity from modified acceleration
def leap_frog_snd_half(particle):
    next_velocity = particle.v + particle.a * time_unit/2
    return Particle(particle.p, next_velocity, particle.a)
```
