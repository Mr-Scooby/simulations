# File level dependency tree. 

mcpattern.py
├─ imports from helpers.py
│  ├─ make_angle_grid
│  ├─ gaussian_weights
│  ├─ intensity_from_field
│  ├─ random_position
│  └─ random_velocity_thermal
├─ imports from rpattern.py
│  └─ array_factor_general
└─ imports from rplotting.py
   ├─ plot_pattern_3d
   ├─ plot_planar_cuts
   └─ plot_atoms

rpattern.py
├─ imports from helpers.py
│  ├─ make_angle_grid
│  ├─ atom_grid
│  ├─ gaussian_weights
│  ├─ intensity_from_field
│  ├─ random_position
│  └─ random_velocity_thermal
└─ imports from rplotting.py
   ├─ plot_pattern_3d
   ├─ plot_planar_cuts
   └─ plot_atoms

helpers.py
└─ no local project imports

rplotting.py
└─ no local project imports


# function level dependencies
## helpers.py 
make_angle_grid
└─ no local function dependencies

single_dipole_E
└─ no local function dependencies

intensity_from_field
└─ single_dipole_E

atom_grid
└─ no local function dependencies

random_position
└─ no local function dependencies

random_velocity_thermal
└─ no local function dependencies

gaussian_weights
└─ no local function dependencies

## rpattern.py
array_factor_general
└─ no local project function calls

centered_indices
└─ no local project function calls

array_factor_separable
└─ centered_indices

get_I_at
└─ no local project function calls

sanity_printing
└─ get_I_at

## mcpattern
positions_at_time
└─ no local project function calls

array_factor_general_time
├─ positions_at_time
└─ rpattern.array_factor_general

make_weight_fn_gaussian_beam
└─ helpers.gaussian_weights

mc_intensity_time_series
├─ helpers.random_position
├─ helpers.random_velocity_thermal
├─ mcpattern.array_factor_general_time
└─ helpers.intensity_from_field

main
├─ helpers.make_angle_grid
├─ mcpattern.mc_intensity_time_series
├─ helpers.random_position
├─ helpers.random_velocity_thermal
├─ helpers.gaussian_weights
├─ rplotting.plot_atoms
├─ rplotting.plot_planar_cuts
├─ rplotting.plot_pattern_3d
└─ mcpattern.make_weight_fn_gaussian_beam

## rplotting.py 
plot_pattern_3d
└─ no local function dependencies

_wrap_to_pi
└─ no local function dependencies

_nearest_index_periodic
└─ _wrap_to_pi

_nearest_index
└─ no local function dependencies

plot_planar_cuts
├─ _nearest_index
├─ _nearest_index_periodic
└─ _wrap_to_pi   (indirectly through helper use)

plot_atoms
└─ no local project function dependencies

