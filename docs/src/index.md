# Homogenization.jl

Consider the problem $$-\nabla \cdot a(x) \nabla u + \lambda u = f$$ in 2D and 3D where $a$ is a symmetric matrix with piecewise constant coefficients in the domain $U \subset \mathbb{R}^{d}$ with $d = 2, 3$.

## Base mesh
We will assume it is feasible to make a triangulation of $U$ such that $a$ is constant on each triangle or tetrahedron. For this we need a mesh

```@docs
Mesh
```

There are some helper functions as well to generate a simple mesh

```@docs
hypercube
```

Now we generate random conductivity parameters for each element of the base mesh.

```julia
using Homogenization
using Homogenization: conductivity_per_element, generate_conductivity

n = 32
base = hypercube(Tri{Float64}, n)
a = conductivity_per_element(base, generate_conductivity(base, n))
```

Next we could try to visualize the base mesh by exporting it to a file compatible
with [Paraview](https://www.paraview.org/). We store the value of `a` as vectorial
data for each element of the base mesh:

```julia
using WriteVTK
using Homogenization: dimension
vtk_grid("checkerboard", base) do vtk
    as_matrix = reshape(reinterpret(Float64, a), dimension(base), :)
    vtk_cell_data(vtk, as_matrix, "a")
end
```

which saves a file `checkerboard.vtu` in the current working directory. It should
look like

![Paraview screenshot](paraview.png)

## Implicit refinement

In FEM we wish to do $h$-refinement of this
triangulation, but the fully refined grid is assumed to be too large to store explicitly.

