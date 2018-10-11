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
function more_testing(n::Int = 5, ahom = 3.0)

    # Number of refinements of the very coarse grid
    m = 2

    # Number of subsequent refinements to intermediate grid where conductivities are assigned
    k = n - m

    # Number of refinements of the intermediate grid.
    l = 3

    # Create a very coarse grid that serves as the coarsest grid for the homogenized operator
    very_coarse_base = hypercube(Tri{Float64}, 2^m, scale = 2^k, origin = (1,1,1), sorted = false)

    # Then refine that grid to a total of n times, and assign a conductivity to each of these elements
    finer_base = refine_uniformly(very_coarse_base, times = k)
    σs = conductivity_checkerboard(very_coarse_base, n)
    σ_el = conductivity_per_element(finer_base, σs)
    
    # Construct coarse grid operator
    interior = list_interior_nodes(very_coarse_base)
    F = cholesky(assemble_matrix(very_coarse_base, (∇u, ∇v) -> ahom * dot(∇u, ∇v))[interior,interior])
    base_level = BaseLevel(Float64, F, nnodes(very_coarse_base), interior)

    # Construct the implicit grid and do the connectivity bookkeeping
    @info "Building implicit grid"
    total_grids = k + l + 1 # with k + l refinements there are k + l + 1 grids!
    implicit = ImplicitFineGrid(very_coarse_base, total_grids) 
    nodes, edges, faces = list_boundary_nodes_edges_faces(implicit.base)
    constraint = ZeroDirichletConstraint(nodes, edges, faces)
    @info "Built!" implicit

    # We build the λI - ∇⋅σ∇ operator only on the finest grid with λ = 0.
    @info "Building diffusion operators"
    A_fine_op = L2PlusDivAGrad(_build_local_diffusion_operators(implicit.reference.levels[end]), mass_matrix(implicit.reference.levels[end]), constraint, 0.0, σ_el)

    # Then we build the homogenized operator -ahom*Δ on all levels.
    A_hom_ops = map(build_local_diffusion_operators(implicit.reference)) do op
        SimpleDiffusion(op, constraint, ahom)
    end

    # Next we allocate the solution vector x, the right-hand side b and the
    # residual vector r on every level.
    @info "Allocating state vectors x, b and r"
    level_states = map(1 : total_grids) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    # Then we initialize the problem on the finest level by
    # setting up a random vector x and a trivial rhs: -∇.a∇u = 1
    @info "Initializing a random x with zero b.c."
    rand!(level_states[total_grids].x)
    broadcast_interfaces!(level_states[total_grids].x, implicit, total_grids)
    apply_constraint!(level_states[total_grids].x, total_grids, constraint, implicit)

    @info "Constructing a local right-hand side"
    local_rhs!(level_states[total_grids].b, implicit)

    ωs = fill(0.001, total_grids)

    for i = 1 : 10
        vcyle_with_ahom!(implicit, base_level, A_fine_op, A_hom_ops, level_states, ωs, total_grids, 10)

        # Check the residual norm
        zero_out_all_but_one!(level_states[total_grids].r, implicit, total_grids)
        @show norm(level_states[total_grids].r)

        vtk_grid("pde_$i", construct_full_grid(implicit, k)) do vtk
            vtk_point_data(vtk, level_states[total_grids].x[1 : nnodes(refined_mesh(implicit, k)), :][:], "x")
        end
    end

    vtk_grid("wut", finer_base) do vtk
        vtk_cell_data(vtk, σ_el, "conductivity")
    end
end

function vcyle_with_ahom!(implicit, base_level, A_fine_op, A_hom_ops, levels, ωs, k, steps)
    # Smooth with the -∇⋅a∇ operator
    for i = 1 : steps
        smoothing_step!(implicit, A_fine_op, ωs[k], levels[k], k)
    end

    # Do a 'coarse grid' solve with the homogenized operator.
    vcycle!(implicit, base_level, A_hom_ops, levels, ωs, k, 2)

    # Smooth with the -∇⋅∇ operator
    for i = 1 : steps
        smoothing_step!(implicit, A_fine_op, ωs[k], levels[k], k)
    end
end

conductivity_checkerboard(mesh::Mesh{3}, n) = [rand() < 0.5 ? 9.0 : 1.0 for i = 1:2^n, j=1:2^n, k=1:2^n]
conductivity_checkerboard(mesh::Mesh{2}, n) = [rand() < 0.5 ? 9.0 : 1.0 for i = 1:2^n, j=1:2^n]

function conductivity_per_element(mesh::Mesh{dim}, σ::Array{Tv,dim}) where {Tv,dim}
    σ_el = Vector{Tv}(undef, nelements(mesh))

    for (idx, el) in enumerate(mesh.elements)
        indices = unsafe_trunc.(Int, mean(get_nodes(mesh, el))).data
        σ_el[idx] = σ[CartesianIndex(indices)]
    end

    σ_el
end