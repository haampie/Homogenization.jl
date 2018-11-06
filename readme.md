# Homogenization.jl

| **Documentation**           | **Build Status**                                                                     |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| [![](docs-badge)](docs-url) | [![Build Status](travis-badge)](travis-url) [![codecov](codecov-badge)](codecov-url) |

This package provides some tools to do large-scale homogenization of elliptic PDEs in 2D and
3D.

Features include:
1. Support for very fine grids for finite elements without storing the grid / matrix 
   explicitly -- should just work on a laptop or workstation computer.
2. A geometric multigrid solver
3. An algorithm to approximate the homogenized coefficients of an elliptic operator with 
   piece-wise continuous coefficients

## Getting started

A recommended way to work with the package now is to clone the package via the package 
manager in a development mode.

1. Open Julia 1.0 in a terminal
2. Hit `]` to open the package manager
3. Enter `dev https://github.com/haampie/Homogenization.jl` or `dev git@github.com:haampie/Homogenization.jl.git`

This will copy the latest version of the code to `~/.julia/dev/Homogenization.jl`.

To verify things work, run the tests via `] test Homogenization`.

### Workflow tips

**Editor.** An editor tailored for Julia is [Juno](http://junolab.org/). Also 
[Visual Studio Code](https://code.visualstudio.com/) with a [Julia extension](https://marketplace.visualstudio.com/items?itemName=julialang.language-julia) is good.

**Development.** It is very convenient to install `Revise.jl` to get a better development 
and testing experience in Julia. This package will do the minimal recompilation when a 
function in the package is changed. 

1. Install it via `] add Revise`.
2. Then in a standard Julia terminal run `using Revise`
3. Only then write `using Homogenization`.

**Threading.** To make use of threading and some more optimized code, one can start Julia
in the terminal as follows:

```
JULIA_NUM_THREADS=N julia -O3
```

here `N` is the number of threads avaiable -- it is best to set it to the number of cores
on the machine.

## Visualization

The package uses WriteVTK.jl to output visualizations that can be viewed in [Paraview](https://www.paraview.org/).

## Example usage

To run the 2D or 3D checkerboard example, try

```julia
using Homogenization

# 2D
ahom_checkerboard(64 + 2 * 10, Tri{Float64}; boundary_layer = 10, refinements = 3, tol = 1e-4, k_max = 3, smoothing_steps = 2, save = 1)

# 3D
ahom_checkerboard(32 + 2 * 10, Tet{Float64}; boundary_layer = 10, refinements = 2, tol = 1e-4, k_max = 3, smoothing_steps = 2, save = 1)
```

This will output a lot of data about the intermediate steps of multigrid and will save the
intermediate approximate solutions of the recurrence `v₀`, `v₁`, ... to separate
files `ahom_0.vtu`, `ahom_1.vtu`, ... in the current working directory. It also creates a 
file `checkerboard.vtu` with the coefficient field. To not save these files use `save = 0`. 
Open the files in Paraview for a visualization.

[travis-badge]: https://travis-ci.org/haampie/Homogenization.jl.svg?branch=master
[travis-url]: https://travis-ci.org/haampie/Homogenization.jl
[docs-badge]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://haampie.github.io/Homogenization.jl/dev
[codecov-badge]: https://codecov.io/gh/haampie/Homogenization.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/haampie/Homogenization.jl