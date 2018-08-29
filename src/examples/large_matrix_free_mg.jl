#=
We wanna solve problems -∇⋅a(x)∇u = f with zero b.c. on large domains Ω = [0,n]ᵈ
for d = 2 or 3. We assume some checkerboard-like structure s.t. a(x) is constant
in z + [0,1] for z ∈ ℤᵈ. A grid is build by uniformly refining the uniform grid
of simplices with nodes on integer coordinates (so h = 1 initially).

The main problem is that `n` is assumed to be so large that a coarse grid
with grid points in integer coordinates is already too large.
In 3D we assume we cannot factorize the sparse matrix of order O(n³) easily.
In practice this already happens around `n = 60` I guess; it would be nice to
have say 128 cells per dimension, which would correspond to a coarse grid
of 2M unknowns. Or even more.

The way to solve this problem is with two tricks:
1. We apply the operator -∇⋅a(x)∇ only on the finest grid in a matrix-free
   fashion
2. Then the coarse grid correction is done with the homogenized operator
   -∇⋅ā∇ which is just scalar.
This helps, because projecting the variable coefficient operator -∇⋅a(x)∇ to
a coarser grid than integer coordinates requires a work, storage and yields
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
    base_tets_ℤ³_compatible_upon_refinement(m) -> Mesh

Returns a cube [1, m+1]^3 split into 6 tets, such that lg(m) uniform refinements
are possible leading to a uniform grid with nodes on the integer coordinates if
m = 2^k for some integer k.
"""
function base_tets_ℤ³_compatible_upon_refinement(m)
    nodes = SVector{3,Float64}[
        (0,0,0), (1,0,0),
        (1,1,0), (0,1,0),
        (0,0,1), (1,0,1),
        (1,1,1), (0,1,1),
    ] .* 2^m .+ 1

    # Specific & magic order that makes sure there's no tets crossing ℤ
    elements = NTuple{4,Int}[
        (1,2,3,7), (1,3,7,4),
        (1,2,6,7), (1,4,7,8),
        (1,5,7,8), (1,5,7,6)
    ]
    return Mesh(nodes, elements)
end

function some_testing(n)
    base = base_tets_ℤ³_compatible_upon_refinement(n)
    fine = refine_uniformly(base, times = n)
    σs = [rand() < 0.5 ? 9.0 : 1.0 for i = 1:2^n, j=1:2^n, k=1:2^n]
    σ_el = conductivity_per_element(fine, σs)

    vtk_grid("wut", fine) do vtk
        vtk_cell_data(vtk, σ_el, "conductivity")
        vtk_cell_data(vtk, 1:nelements(fine), "ss")
    end
end

function more_testing(n)
    base = hypercube(Tet{Float64}, 1, scale = 2^n, origin = (1,1,1))
    fine = refine_uniformly(base, times = n)
    σs = [rand() < 0.5 ? 9.0 : 1.0 for i = 1:2^n, j=1:2^n, k=1:2^n]
    σ_el = conductivity_per_element(fine, σs)

    vtk_grid("wut", fine) do vtk
        vtk_cell_data(vtk, σ_el, "conductivity")
        vtk_cell_data(vtk, 1:nelements(fine), "ss")
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