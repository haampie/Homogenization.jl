# Homogenization.jl

This package provides some tools to do large-scale homogenization of elliptic PDEs in 2D and
3D.

Features include:
1. Support for very fine grids for finite elements without storing the grid / matrix 
   explicitly.
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



## Visualization

The package uses WriteVTK.jl to output visualizations that can be viewed in [Paraview](https://www.paraview.org/).