# File Map

## `helpers.py`
Reusable utilities shared across the project.

### Main role
Provides low-level support functions that do not define the main simulation workflow by themselves.

### Typical contents
- angle-grid creation
- atom position generation
- thermal velocity generation
- Gaussian beam weighting
- conversion from field-like quantities to intensity

### Put code here when
- the function is small and reusable
- it supports multiple modules
- it is not specifically plotting logic
- it is not specifically Monte Carlo orchestration
- it is not the main array-factor formula itself

---

## `rpattern.py`
Static radiation-pattern and array-factor calculations.

### Main role
Contains the core formulas that compute the array factor or related pattern quantities for a fixed set of atom positions and weights.

### Typical contents
- general array-factor computation
- separable or lattice-based array-factor formulas
- physics calculations directly tied to the radiation pattern

### Put code here when
- the function computes the array factor or a closely related pattern quantity
- the function belongs to the static or single-configuration calculation
- the function is part of the core physics model

### Avoid putting here
- plotting code
- Monte Carlo loops
- standalone demo code that mixes setup, computation, and plotting

---

## `mcpattern.py`
Time-dependent Monte Carlo simulation for moving atoms.

### Main role
Coordinates the stochastic simulation workflow for many realizations and time steps.

### Typical contents
- position updates in time
- time-dependent array-factor wrappers
- Monte Carlo averaging loops
- per-realization sampling and accumulation logic

### Put code here when
- the function manages repeated realizations
- the function evolves atom positions in time
- the function averages outcomes over many sampled worlds
- the function belongs to the simulation driver rather than the base formula

### Avoid putting here
- general plotting utilities
- static helper utilities already reusable elsewhere
- the low-level static array-factor formula itself

---

## `rplotting.py`
Visualization utilities.

### Main role
Contains functions that only display or visualize data.

### Typical contents
- 3D radiation-pattern plots
- planar cut plots
- atom-cloud visualization
- arrows for dipole direction, incident wave direction, and velocities

### Put code here when
- the output is a figure or axis
- the function is purely visual
- the function formats or presents results without changing the model

### Avoid putting here
- simulation logic
- array-factor physics
- random sampling logic

---

## Practical navigation guide
When you are unsure where something belongs, use this rule:

- If it computes the pattern physics directly, use `rpattern.py`.
- If it samples or supports several modules, use `helpers.py`.
- If it performs time evolution or Monte Carlo averaging, use `mcpattern.py`.
- If it creates figures, use `rplotting.py`.

---

## Suggested future entry scripts
To keep the library files clean, runnable examples are best kept in separate scripts such as:

- `run_static.py` for a fixed-pattern example
- `run_mc.py` for the Monte Carlo time-dependent example

That keeps the core modules focused on one responsibility each.
