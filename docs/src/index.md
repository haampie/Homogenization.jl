# Homogenization.jl

Consider the problem $$-\nabla \cdot a(x) \nabla u + \lambda u = f$$ in 2D and 3D where $a$ is a symmetric, positive definite matrix with piecewise constant coefficients in the domain $U \subset \mathbb{R}^{d}$ with $d = 2, 3,$ $\lambda > 0$ and $u = 0$ on $\partial U$.

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

## Solving a "coarse" FEM problem

For the moment let's stick to a very coarse grid for the checkerboard as we have it with
just four nodes on the corners of each checkerboard cell (one node per cell). The weak form reads 

$$\int_U a \nabla u \cdot \nabla v + \lambda uv = \int fv.$$

for $u, v$ in the appropriate space, which we discretize by taking piece-wise linear elements for $u$ and $v.$ Let's take
$f = \lambda = 1.$

This is done as follows:

```julia
using Homogenization: assemble_checkerboard, assemble_vector
A = assemble_checkerboard(base, a, 1.0)
b = assemble_vector(base, identity)
```

The above does not take into account that $u$ is supposed to be zero at the boundary.
To impose the boundary condition we have to detect the boundary in the mesh first. 
This is rather simple:

```julia
using Homogenization: list_interior_nodes
interior = list_interior_nodes(base)
```

The boundary condition is simply imposed by partitioning the unknowns $x = \begin{bmatrix}x_i & x_b\end{bmatrix}^T$ such that the linear system reads:

$$\begin{bmatrix}A_{ii} & A_{ib} \\ A_{bi} & A_{bb}\end{bmatrix}\begin{bmatrix}x_i \\ x_b\end{bmatrix} = \begin{bmatrix}b_i \\ b_b\end{bmatrix}.$$ With the boundary nodes $x_b = 0$ this comes down to solving $A_{ii}x_i = b_i:$

```julia
using Homogenization: nnodes
A_int = A[interior, interior]
b_int = b[interior]
x_int = A_int \ b_int
x = zeros(nnodes(base))
x[interior] .= x_int
```

Finally we would like to visualize the FEM solution in Paraview, which is done as follows:

```julia
vtk_grid("checkerboard", base) do vtk
    as_matrix = reshape(reinterpret(Float64, a), dimension(base), :)
    vtk_cell_data(vtk, as_matrix, "a")
    vtk_point_data(vtk, x, "x")
end
```

And it will look more or less like

![Paraview solution](solution.png)

## Implicit refinement

The coarse FEM problem will have a large error as the coefficients are rough and the number
of nodes per checkerboard cell is approximately one. Therefore we wish to do $h$-refinement.
The assumption will be that we will not be able to store the fully refined grid (nodes and
elements) and neither the corresponding discrete operator (`A_int`).

It makes sense to try multigrid to get $h$-refinement