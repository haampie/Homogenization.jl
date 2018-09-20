#=
We wanna solve problems -∇⋅a(x)∇u = f with zero b.c. on large domains Ω = [0,n]ᵈ
for d = 2 or 3. We assume some checkerboard-like structure s.t. a(x) is constant
in z + [0,1] for z ∈ ℤᵈ. A grid is build by uniformly refining the uniform grid
of simplices with nodes on integer coordinates (so h = 1 initially).

The main problem is that `n` is assumed to be so large that a coarse grid
with grid points in integer coordinates is already too large.
In 3D we assume we cannot factorize the sparse matrix of order O(n³) easily.
In practice this already happens around `n > 64` I guess; it would be nice to
have say 128 cells per dimension, which would correspond to a coarse grid
of 2M unknowns. Or even more.

The way to solve this problem is with two tricks:
1. We apply the operator -∇⋅a(x)∇ only on the finest grid in a matrix-free
   fashion
2. Then the coarse grid correction is done with the homogenized operator
   -∇⋅ā∇ which is just scalar.
This helps, because projecting the variable coefficient operator -∇⋅a(x)∇ to
a coarser grid than integer coordinates requires work, storage and yields
probably a bad coarse operator. With this scheme the ā thing has constant
coefficients and we can construct the coarse grid operators without having to
create Galerkin-projections from the finest grid. Secondly, the size of the
coarse grid for problem 2 is not bounded below by the number of cells `n`, so
we can coarsen a few steps more.

How to implement it:
1. We're just gonna assume that n = 2
=#

using Random

"""
    more_testing(n)

Create a coarse grid of size 2^n in 3 dimensions.

Example setup:
n = 6 → coarse grid operator is still factorizable with the memory we have.
n = 7 → out of memory, coarse base mesh has 2_146_689 nodes and 12_582_912 
        elements.

Now we can refine this (pretty fine) coarse mesh a few times:

+-------+----------------+----------------+----------+
| #refs | # nodes stored | # problem size | overhead |
+-------+----------------+----------------+----------+
|    0x |     50_331_648 |      2_146_689 |    23.5x | 
|    1x |    125_829_120 |     16_974_593 |     7.5x |
|    2x |    440_401_920 |    135_005_697 |     3.2x |
|    3x |  2_076_180_480 | ~1_000_000_000 |     2.0x |
--------+----------------+----------------+----------+

For the implicit grid approach we need to store nelements(base) * nnodes(fine),
so there is some overhead. On the cube: every nodes is connected to about 6
elements, and each element has 4 nodes, so every nodes is stored about 24 times
with the implicit grid approach. But then after refinement, the number of shared
nodes drops rapidly. Two refinements has an overhead of ~3x, which is OK.

Also 2x refinement is probably the best we can get judging from the number of 
nodes.
"""
function more_testing(n)
    @time very_coarse_base = hypercube(Tet{Float64}, 1, scale = 2^n, origin = (1,1,1))
    @time finer_base = refine_uniformly(very_coarse_base, times = n)
    @time σs = [rand() < 0.5 ? 9.0 : 1.0 for i = 1:2^n, j=1:2^n, k=1:2^n]
    @time σ_el = conductivity_per_element(finer_base, σs)
    @time refined_reference = refined_element(2, typeof(finer_base))

    return
    # return nelements(finer_base) .* [nnodes(lvl) for lvl in refined_reference.levels]

    @show nnodes(finer_base) nelements(finer_base)

    vtk_grid("wut", finer_base) do vtk
        vtk_cell_data(vtk, σ_el, "conductivity")
    end
end

function conductivity_per_element(mesh::Mesh{dim}, σ::Array{Tv,dim}) where {Tv,dim}
    σ_el = Vector{Tv}(undef, nelements(mesh))

    for (idx, el) in enumerate(mesh.elements)
        indices = unsafe_trunc.(Int, mean(get_nodes(mesh, el))).data
        σ_el[idx] = σ[CartesianIndex(indices)]
    end

    σ_el
end