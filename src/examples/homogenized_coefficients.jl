using Statistics: mean

export checkerboard_homogenization


"""
Returns the size of the boundary layer for the operator `λ - ∇⋅A∇`.
"""
compute_boundary_layer(λ::Float64) = floor(Int, 5 * max(1.0, λ^-0.5) * max(1.0, log2(λ^-0.5)))
compute_box_radius(k::Int, n::Int, ε::Float64 = 0.0) = floor(Int, 2 ^ (n - k * (0.5 - ε)))

infnorm(x::SVector{2}) = max(abs(x[1]), abs(x[2]))
infnorm(x::SVector{3}) = max(abs(x[1]), abs(x[2]), abs(x[3]))

elementcenter(mesh, el) = mean(get_nodes(mesh, el))

"""
Makes a copy of the mesh with nodes and elements sorted by distance to the origin in
infnorm (elements sorted by the center).
"""
function order_nodes_and_elements_by_magnitude(mesh::Mesh{dim,N}) where {dim,N}
    I = sortperm(mesh.nodes, by = infnorm); J = invperm(I)
    sorted_mesh = Mesh(mesh.nodes[I], map(mesh.elements) do element
        return sort_bitonic(ntuple(i -> @inbounds(J[element[i]]), N))
    end)
    sort!(sorted_mesh.elements, by = el -> infnorm(elementcenter(sorted_mesh, el)))
    return sorted_mesh
end

"""
Return a range `r` such that `mesh.elements[r]` all lie within a distance `radius` from
the origin in infnorm. This requires that the elements are sorted.
"""
find_elements_in_radius(mesh::Mesh, radius) = OneTo(searchsortedlast(
    mesh.elements, 
    radius, 
    lt = (dist, el) -> dist < infnorm(elementcenter(mesh, el))
))

"""
Return a range `r` such that `mesh.nodes[r]` all lie within a distance `radius` from
the origin in infnorm. This requires that the nodes are sorted.
"""
find_nodes_in_radius(mesh::Mesh, radius) = OneTo(searchsortedlast(
    mesh.nodes,
    radius + 10eps(),
    lt = (dist, node) -> dist < infnorm(node)
))

"""
When the domain shrinks we also drop all the nodes in the state vectors that lie outside
the new domain. In principle one can avoid the copy here by working with views.
"""
shrink_level_state(l::LevelState{Tv,V}, elements) where {Tv,V} = LevelState{Tv,V}(
    l.x[:, elements],
    l.b[:, elements], 
    l.r[:, elements], 
    l.p[:, elements], 
    l.Ap[:, elements]
)

function random_unit_vec(::Type{<:ElementType{dim}}) where {dim}
    ξ = @SVector(ones(dim))
    return ξ / norm(ξ)
end


"""
Saves the conductivity info to a file named checkerboard.vtu
"""
function export_domain(base::Mesh{dim}, cond::Vector) where {dim}
    @info "Saving the grid checkerboard grid"
    vtk_grid("checkerboard", base) do vtk
        # WriteVTK.jl wants us to deliver a matrix of size dim × nelements for 
        # vectorial data, so we just reshape and reinterpret the vector of static 
        # vectors `cond`.
        vtk_cell_data(vtk, reshape(reinterpret(Float64, cond), dim, :), "a")
    end
end

function export_unknown(base::Mesh{dim}, implicit::ImplicitFineGrid, x::AbstractMatrix, k::Int, level::Int) where {dim}
    vtk_grid("ahom_$k", construct_full_grid(implicit, level)) do vtk
        # extract the nodal values of `x` on implicit grid number `save` and 
        # enumerate them as a long vector.
        vtk_point_data(vtk, x[1 : nnodes(refined_mesh(implicit, level)), :][:], "v")
    end
end

"""
    checkerboard_homogenization(n::Int = 4, type::Type = Tri64;
        refinements::Int = 2,
        smoothing_steps::Int = 3,
        tolerance::Float64 = 1e-4,
        ξ::SVector = random_unit_vec(type),
        save::Union{Nothing,Int} = nothing
    ) → σ

Implements a continuous version of "Efficient methods for the estimation of homogenized 
coefficients" (https://arxiv.org/abs/1609.06674 section 11. Numerical tests).

The domain is a hypercube [-2^n,2^n]ᵈ on which a checkerboard pattern is generated with
unit size length per cell. The operator we want to homogenize is of the form 
`L = -∇⋅A∇` where `A = diag(a_1(x), ..., a_d(x))` and we set `a_i` to be constant in each
checkerboard cell with values 1 or 9 with equal odds.

One can approximate the values of the homogenized operator `Lhom = L = -∇ ⋅ Ahom∇` as 
follows:

1. Compute the expected value E of ξ⋅Ahom ξ (in our case simply ½ * 1 + ½ * 9 = 5), 
2. Fix a unit vector `ξ = @SVector [1.0, 0.0, 0.0]`
3. Run this function to obtain a correction `σ` to the expected value
4. Compute the coefficient ξ⋅Ahom ξ ≈ E - σ

Since this function only applies the algorithm to a single generated domain, step 3 and 4
are to be repeated.

ELEMENT TYPES. This function works both in 2D and 3D -- just set `type = Tri64` for 2D and 
`type = Tet64` for 3D.

REFINEMENT. To reduce the FEM error, one can refine the grid a few times. Without 
refinements each checkerboard cell is split in two triangles (2D) or five tetrahedra (3D).
One level of refinement splits a triangle into four smaller triangles and a tetrahedron into
eight smaller tetrahedra. By default `refinements = 2`.

BOUNDARY LAYER. We have to set an artificial zero boundary condition of the domain. The 
boundary layer / layer of influence of this b.c. is hard-coded in the 
`compute_boundary_layer` function.

SOME IMPLEMENTATION DETAILS. The algorithm has an 'outer iteration' (over `k` in the 
article) where `vₖ` is used to compute increments to `σ` and the domain is shrunk; and 
an 'inner iteration' where `vₖ` is approximately computed with multigrid.

The outer iteration is stopped `after` n steps OR (more likely) whenever the boundary
layer grows faster than the domain shrinks.

For the coarsest grid of multigrid see REFINEMENT. Conjugate gradients is used as a smoother
for multigrid, but it is only approximate due to the way we store the unknowns: nodes on the
boundaries of checkerboard cells are stored multiple times. The three dot products in each
step of CG do not take this into account, so there might be a slight error.

Multigrid accepts an approximate solution whenever the increment to σ is below a certain
threshold.

To make use of a multithreaded matrix-vector in multigrid, start Julia using
`JULIA_NUM_THREADS=n julia -O3` where `n` is the number of threads (typically 2, 4 or 8).

EXPORTING. To see the generated checkerboard and the intermediate vₖ's, set the `save`
parameter to the level of refinement that you want to save. E.g. `vₖ = 1` saves the coarse
grid, `save = 2` the coarse grid after one refinement step, etc. By default no data is
exported (which is the setting `save = nothing`).

Example 2D -- initial domain size w/ boundary layer [-37,37]^2:
```
julia> # One refinement: Domain size = [-37,37]^2
julia> checkerboard_homogenization(5, Tri64, refinements = 1, tolerance = 1e-5)
1.6163911040833774
julia> checkerboard_homogenization(5, Tri64, refinements = 2, tolerance = 1e-5)
1.8862838217833766
julia> checkerboard_homogenization(5, Tri64, refinements = 3, tolerance = 1e-5)
1.9454383432630586
```
Example 3D -- initial domain size w/ boundary layer [-13,13]^3
```
julia> checkerboard_homogenization(3, Tet64, refinements = 1, tolerance = 1e-4)
0.7989162402285056
julia> checkerboard_homogenization(3, Tet64, refinements = 2, tolerance = 1e-4)
1.0629164417822408
julia> checkerboard_homogenization(3, Tet64, refinements = 3, tolerance = 1e-4)
1.223149465555829
```
"""
function checkerboard_homogenization(
    n::Int = 4,
    type::Type{ElT} = Tri64;
    refinements::Int = 2,
    smoothing_steps::Int = 3,
    tolerance::Float64 = 1e-4,
    ξ::SVector = random_unit_vec(type),
    save::Union{Nothing,Int} = nothing
) where {dim,N,Tv,ElT<:ElementType{dim,N,Tv}}

    # The coefficient of the big L2-term
    λ = 1.0

    # The correction to the averaged/expected coefficient
    σ = 0.0

    # This is the domain which we integrate over
    box_radius = compute_box_radius(0, n)

    # This is growing boundary layer where the solution is affected by the artificial 
    # zero boundary condition
    boundary_layer = compute_boundary_layer(λ)

    # Half-width of the box of the total domain
    total_radius = box_radius + boundary_layer

    # The hyperube should be centered at the origin
    shift = @SVector fill(total_radius, dim)

    # Generate a mesh for the square or cube [-total_radius, total_radius]^d
    # where the nodes are ordered by distance to the origin
    base = order_nodes_and_elements_by_magnitude(hypercube(
        ElT, 
        2 * total_radius,
        origin = -shift)
    )

    # We generate an array with random conductivity coefficients and map them to the mesh
    cond = conductivity_per_element(
        base, 
        generate_conductivity(base, 2 * total_radius),
        shift .+ 1
    )

    # Maybe export the domain to visualize in paraview
    save !== nothing && export_domain(base, cond)

    # Build the implicit grid
    total_grids = refinements + 1
    implicit = ImplicitFineGrid(base, total_grids)

    # Set up the zero boundary condition
    nodes, edges, faces = list_boundary_nodes_edges_faces(base)
    constraint = ZeroDirichletConstraint(nodes, edges, faces)

    diff_terms = build_local_diffusion_operators(implicit.reference)
    mass_terms = build_local_mass_matrices(implicit.reference)
    level_operators = map(zip(diff_terms, mass_terms)) do op
        diff, mass = op
        L2PlusDivAGrad(diff, mass, constraint, λ, cond)
    end

    # Allocate state vectors x, b and r on all levels.
    level_states = map(1 : total_grids) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    # Allocate previous v
    v_prev = similar(level_states[end].x)

    # @info "Initializing a random x with zero b.c."
    rand!(level_states[end].x)
    broadcast_interfaces!(level_states[end].x, implicit, total_grids)
    apply_constraint!(level_states[end].x, total_grids, constraint, implicit)

    # @info "Building the initial local r.h.s."
    ∂ϕ∂xᵢs = partial_derivatives_functionals(refined_mesh(implicit, total_grids))
    rhs_aξ∇v!(level_states[end].b, ∂ϕ∂xᵢs, implicit, cond, ξ)

    for k = 0 : n

        @info "Step $k. Domain size = [-$total_radius,$total_radius]^$dim:" box_radius boundary_layer implicit

        # Factorize the coarse grid operator
        interior = list_interior_nodes(base)
        F = cholesky(assemble_checkerboard(base, cond, λ)[interior,interior])
        base_level = BaseLevel(Float64, F, nnodes(base), interior)

        @info "Done building the new coarse grid operator. Now running multigrid."

        Δσ = 0.0
        Δσ_prev = 0.0

        ### Solve for v_k by doing V-cycles until the increment in sigma is sufficiently small
        for i = 1 : 1000
            vcycle!(implicit, base_level, level_operators, level_states, total_grids, smoothing_steps)

            integration_domain_elements = find_elements_in_radius(base, box_radius)
            integration_domain_area = integrate_area(level_operators[end], implicit, integration_domain_elements)

            # Initial rhs is special
            integral = if k == 0
                integrate_first_term(level_states[end].x, ∂ϕ∂xᵢs, implicit, integration_domain_elements, level_operators[end], cond, ξ)
            else
                integrate_terms(level_states[end].x, v_prev, implicit, integration_domain_elements, level_operators[end])
            end

            # Compute the increment to σ
            Δσ = 2^k * integral / integration_domain_area

            # Compute (inexact) residual norm
            zero_out_all_but_one!(level_states[end].r, implicit, total_grids)
            @info "Multigrid step $i" norm(level_states[end].r) σ + Δσ abs(Δσ - Δσ_prev)

            # If there is no more significant contribution to σ we stop multigrid
            abs(Δσ - Δσ_prev) < tolerance && break

            Δσ_prev = Δσ
        end

        σ += Δσ

        ### Shrink the domain
        @info "Shrinking the domain"
        λ /= 2
        box_radius = compute_box_radius(k + 1, n)
        boundary_layer = compute_boundary_layer(λ)

        save !== nothing && export_unknown(base, implicit, level_states[end].x, k, save)

        # Our domain starts growing again, so we just stop here with the iteration
        box_radius + boundary_layer > total_radius && break
        
        # Otherwise shrink the domain!
        total_radius = box_radius + boundary_layer
        shrunken_mesh_nodes = find_nodes_in_radius(base, total_radius)
        shrunken_mesh_elements = find_elements_in_radius(base, total_radius)
        base = Mesh(base.nodes[shrunken_mesh_nodes], base.elements[shrunken_mesh_elements])

        # Set up the new zero boundary condition
        nodes, edges, faces = list_boundary_nodes_edges_faces(base)
        constraint = ZeroDirichletConstraint(nodes, edges, faces)

        # We drop nodes outside the new domain in the state vectors as well
        level_states = map(level_states) do level_state
            shrink_level_state(level_state, shrunken_mesh_elements)
        end

        # And we have to apply the zero b.c. to `x` now
        apply_constraint!(level_states[end].x, total_grids, constraint, implicit)

        # Make a copy of the current x because we have to integrate with it
        v_prev = copy(level_states[end].x)

        # We have to update the operators with the new lambda and new b.c.
        for operator in level_operators
            operator.λ = λ
            operator.constraint = constraint
        end

        # I don't have time to get rid of the redundant work here now :|
        implicit = ImplicitFineGrid(base, total_grids)

        # Finally we have to update the right-hand side
        next_rhs!(level_states[end].b, level_states[end].x, implicit, level_operators[end])
    end

    return σ
end

# There's probably a nicer way to do this, but let's just define
# the weighted dot product like this for 2D and 3D.
@propagate_inbounds function weighted_dot(∇u::SVector{3}, σ::SVector{3}, ∇v::SVector{3})
    ∇u[1] * σ[1] * ∇v[1] + ∇u[2] * σ[2] * ∇v[2] + ∇u[3] * σ[3] * ∇v[3]
end

@propagate_inbounds function weighted_dot(∇u::SVector{2}, σ::SVector{2}, ∇v::SVector{2})
    ∇u[1] * σ[1] * ∇v[1] + ∇u[2] * σ[2] * ∇v[2]
end

"""
Build the operator for the bilinear form B[u,v] = ∫λuv + a∇u⋅∇v.
"""
function assemble_checkerboard(mesh::Mesh{dim,N,Tv,Ti}, σs::Vector{SVector{dim,Tv}}, λ::Tv = 1.0) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = init_values(cell, quadrature, update_gradients | update_det_J)

    total = N * N * nelements(mesh)
    is, js, vs = Vector{Ti}(undef, total), Vector{Ti}(undef, total), Vector{Tv}(undef, total)
    A_local = zeros(N, N)

    idx = 1
    @inbounds for (e_idx, element) in enumerate(mesh.elements)
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))
        σ = σs[e_idx]

        # For each quad point
        @inbounds for qp = 1 : nquadpoints(quadrature), i = 1:N, j = 1:N
            u = get_value(element_values, qp, i)
            v = get_value(element_values, qp, j)
            ∇u = get_grad(element_values, i)
            ∇v = get_grad(element_values, j)
            A_local[i,j] += weights[qp] * (λ * u * v + weighted_dot(∇u, σ, ∇v))
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:N, j = 1:N
            value = A_local[i,j]
            value == zero(Tv) && continue
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = value * get_det_jac(element_values)
            idx += 1
        end
    end

    resize!(is, idx - 1)
    resize!(js, idx - 1)
    resize!(vs, idx - 1)

    # Build the sparse matrix
    return sparse(is, js, vs, nnodes(mesh), nnodes(mesh))
end

"""
Construct the functionals ∫∂ϕ/∂xᵢ for i = 1 : 3
"""
function partial_derivatives_functionals(mesh::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)

    bs = zeros(Tv,dim,nnodes(mesh))

    b_local = zeros(dim, N)

    idx = 1
    @inbounds for (e_idx, element) in enumerate(mesh.elements)
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(b_local, zero(Tv))

        # For each quad point
        @inbounds for qp = 1 : nquadpoints(quadrature), i = 1:N
            ∇v = get_grad(element_values, i)

            for j = 1:dim
                b_local[j, i] += weights[qp] * ∇v[j]
            end
        end

        detJ = get_det_jac(element_values)

        # Copy the local vecs over to the global one
        @inbounds for i = 1:N, j = 1:dim
            bs[j,element[i]] += b_local[j, i] * detJ
        end
    end

    return collect(reshape(reinterpret(SVector{dim,Tv}, bs), :))
end

"""
The initial right-hand side is formed via functional F(v) = -∫aξ⋅∇v. As a is
constant, we just compute ∫∂̂ϕᵢ∂xⱼ on the reference cell, and then use the
pullback to assemble the vector using something like (|J|aξ⋅J⁻¹)∇̂ϕ.
"""
function rhs_aξ∇v!(b::AbstractMatrix, ∂ϕ∂xᵢs::Vector{SVector{dim,Tv}}, implicit::ImplicitFineGrid{dim}, σs::Vector{SVector{dim,Tv}}, ξ::SVector{dim,Tv}) where {dim,Tv}
    # Build a rhs on the finest level of the reference cell
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))

    @assert size(b) == (nnodes(fine), nelements(base))

    cell = cell_type(base)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_inv_J | update_det_J)

    @inbounds for (idx, element) in enumerate(base.elements)
        reinit!(element_values, base, element)

        detJ = get_det_jac(element_values)
        Jinv = get_inv_jac(element_values)
        σ = σs[idx]

        P = -detJ * (Jinv' * (σ .* ξ))

        for i = 1 : nnodes(fine)
            b[i, idx] = dot(∂ϕ∂xᵢs[i], P)
        end
    end
end

struct Conductivity{Tv,N}
    n::Int
    σ::Array{SVector{N,Tv},N}
end

"""
Generate a random (x, y, z) conductance in each unit cube in the domain
of [1, n + 1]^3
"""
generate_conductivity(m::Mesh{3}, n::Int) = Conductivity(n, [rand_cond_3d() for x = 1 : n, y = 1 : n, z = 1 : n])
generate_conductivity(m::Mesh{2}, n::Int) = Conductivity(n, [rand_cond_2d() for x = 1 : n, y = 1 : n])
rand_cond_3d() = SVector{3,Float64}(rand(Bool) ? 1.0 : 9.0, rand(Bool) ? 1.0 : 9.0, rand(Bool) ? 1.0 : 9.0)
rand_cond_2d() = SVector{2,Float64}(rand(Bool) ? 1.0 : 9.0, rand(Bool) ? 1.0 : 9.0)

"""
For convenience this guy will just return a vector `v` s.t. `v[el_idx]` is a
tuple of the conductivity in all spatial directions in that element.
"""
function conductivity_per_element(mesh::Mesh{dim}, σ::Conductivity{Tv,dim}, offset = @SVector(zeros(dim))) where {Tv,dim}
    σ_el = Vector{SVector{dim,Tv}}(undef, nelements(mesh))

    for (idx, el) in enumerate(mesh.elements)
        indices = unsafe_trunc.(Int, mean(get_nodes(mesh, el)) .+ offset).data
        σ_el[idx] = σ.σ[CartesianIndex(indices)]
    end

    σ_el
end


"""
Solve the problem ∇⋅a∇u = 1 in Ω, u = 0 on ∂Ω with a few multigrid steps.
"""
function checkerboard_hypercube_multigrid(n::Int, elementtype::Type{<:ElementType} = Tet{Float64}, refinements = 2, max_cycles = 5, save = 2)
    base = hypercube(elementtype, n)

    # Generate conductivity
    Random.seed!(1)
    σ = generate_conductivity(base, n)
    σ_per_el = conductivity_per_element(base, σ)

    ### Coarse grid.
    interior = list_interior_nodes(base)
    F = cholesky(assemble_checkerboard(base, σ_per_el, 0.0)[interior,interior])
    base_level = BaseLevel(Float64, F, nnodes(base), interior)

    ### Fine grid
    implicit = ImplicitFineGrid(base, refinements)
    nodes, edges, faces = list_boundary_nodes_edges_faces(implicit.base)
    constraint = ZeroDirichletConstraint(nodes, edges, faces)

    @show implicit

    # Build the local operators.
    diff_terms = build_local_diffusion_operators(implicit.reference)
    mass_terms = build_local_mass_matrices(implicit.reference)
    level_operators = map(zip(diff_terms, mass_terms)) do op
        diff, mass = op
        L2PlusDivAGrad(diff, mass, constraint, 0.0, σ_per_el)
    end

    # Allocate state vectors x, b and r on all levels.
    level_states = map(1 : refinements) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    finest_level = level_states[refinements]

    # Initialize a random x.
    rand!(finest_level.x)
    broadcast_interfaces!(finest_level.x, implicit, refinements)
    apply_constraint!(finest_level.x, refinements, constraint, implicit)
    local_rhs!(finest_level.b, implicit)

    rs = Float64[]

    # Solve the next problem
    for i = 1 : max_cycles
        println("Cycle ", i)
        vcycle!(implicit, base_level, level_operators, level_states, refinements, 3)

        # Compute increment in σ and residual norm
        zero_out_all_but_one!(finest_level.r, implicit, refinements)
        push!(rs, norm(finest_level.r))
        @show last(rs)
    end

    full_mesh = construct_full_grid(implicit, save)

    vtk_grid("checkerboard_full_$refinements", full_mesh) do vtk
        vtk_point_data(vtk, finest_level.x[1 : nnodes(refined_mesh(implicit, save)), :][:], "x")
        # vtk_cell_data(vtk, reshape(reinterpret(Float64, σ_per_el), dimension(full_mesh), :), "σ")
    end

    rs
end

function compare_refinements_on_same_material(refinements = 2 : 7)
    results = []
    for ref = refinements
        push!(results, checkerboard(148, Tri{Float64}, ref, 1e-4, 50, 5, 2))
    end
    refinements, results
    #plot(vcat([abs.(results[end][1][i] .- (2results[end][1][i][end] - results[end][1][i][end-1])) for i = 1 : 6]...), yscale = :log10, mark = :o)
    #conv = [sum([results[j][1][i][end] for i = 1 : length(results[j][1])]) for j = 1 : 7]
    #plot([abs.(conv .- (4 * conv[7] - conv[6]) / 3)], yscale = :log10, mark = :x)
end

"""
The initial right-hand side is formed via functional F(v) = -∫aξ⋅∇v. As a is
constant, we just compute ∫∂̂ϕᵢ∂xⱼ on the reference cell, and then use the
pullback to assemble the vector using something like (|J|aξ⋅J⁻¹)∇̂ϕ.

It comes down to summing (linear combination of partial derivatives + M v₀) * v₀
"""
function integrate_first_term(v₀::AbstractMatrix{Tv}, ∂ϕ∂xᵢs::Vector{SVector{dim,Tv}}, implicit::ImplicitFineGrid{dim}, subset::AbstractVector{Ti}, ops::L2PlusDivAGrad, σs::Vector{SVector{dim,Tv}}, ξ::SVector{dim,Tv}) where {dim,Tv,Ti}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_inv_J | update_det_J)

    # Avoid `nelements(base)` allocations
    v₀_local = zeros(Tv, nnodes(fine))
    Mv₀_local = similar(v₀_local)

    total = zero(Tv)

    @inbounds for idx in subset
        reinit!(element_values, base, base.elements[idx])

        detJ = get_det_jac(element_values)
        Jinv = get_inv_jac(element_values)
        σ = σs[idx]
        P = -detJ * (Jinv' * (σ .* ξ))

        # Copy the thing over
        @simd for i = 1 : nnodes(fine)
            v₀_local[i] = v₀[i, idx]
        end

        # Multiply with mass
        mul!(Mv₀_local, ops.mass, v₀_local)

        running_sum = zero(Tv)

        # Inner product
        @simd for i = 1 : nnodes(fine)
            running_sum += v₀_local[i] * (dot(∂ϕ∂xᵢs[i], P) + Mv₀_local[i])
        end

        total += running_sum * get_det_jac(element_values)
    end

    return total
end

function integrate_terms(vₖ::AbstractMatrix{Tv}, vₖ₋₁::AbstractMatrix{Tv}, implicit::ImplicitFineGrid, subset::AbstractVector{Ti}, ops::L2PlusDivAGrad) where {Tv,Ti}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_det_J)

    # Avoid `nelements(base)` allocations
    vₖ_local = zeros(Tv, nnodes(fine))
    Mvₖ_local = similar(vₖ_local)

    total = zero(Tv)

    @inbounds for idx in subset
        reinit!(element_values, base, base.elements[idx])

        # Copy the thing over
        @simd for i = 1 : nnodes(fine)
            vₖ_local[i] = vₖ[i, idx]
        end

        # Multiply with mass
        mul!(Mvₖ_local, ops.mass, vₖ_local)

        # Inner product
        running_sum = zero(Tv)
        @simd for i = 1 : nnodes(fine)
            running_sum += (vₖ[i, idx] + vₖ₋₁[i, idx]) * Mvₖ_local[i]
        end
        total += running_sum * get_det_jac(element_values)
    end

    return total
end

"""
Compute 1ᵗM1 under selected base cells. Maybe it's better to avoid this
computation and incorporate it in the `sum_terms` and `sum_first_term` functions
"""
function integrate_area(ops::L2PlusDivAGrad, implicit::ImplicitFineGrid, subset::AbstractVector{Ti}) where {Ti}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_det_J)

    M_total = sum(ops.mass)
    area = 0.0

    @inbounds for idx in subset
        reinit!(element_values, base, base.elements[idx])
        area += M_total * get_det_jac(element_values)
    end

    area
end

"""
The next *local* right-hand side `b` will simply be `M * x`, mass matrix times
previous solution.
"""
function next_rhs!(b::AbstractMatrix{Tv}, x::AbstractMatrix{Tv}, implicit::ImplicitFineGrid, ops::L2PlusDivAGrad) where {Tv}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_det_J)

    fill!(b, zero(Tv))

    @inbounds for idx = 1 : nelements(base)
        reinit!(element_values, base, base.elements[idx])
        detJ = get_det_jac(element_values)
        offset = (idx - 1) * size(x, 1)

        # b ← λ * |J| * M * x + b
        my_A_mul_B!(ops.λ * detJ, ops.mass, x, b, offset)
    end
    nothing
end

"""
    checkerboard_hypercube_full(n, eltype, refs, λ)

Solve the problem (λ-∇⋅a∇)u₁ = 1 and (λ-∇⋅ā∇)u₂ = 1 using a direct method and
save them to a file `checkerboard_full`. Take for instance a not too large
domain with two refinements in 3D, then ahom ≈ 3.94 according to the docstring
of [`ahom_checkerboard`](@ref).

```
checkerboard_hypercube_full(20, Tet{Float64}, 2, 0.0, 3.94)
```

In Paraview one can compare the two solutions with the Plot over Line tool.
"""
function checkerboard_hypercube_full(n::Int, elementtype::Type{<:ElementType} = Tet{Float64}, refinements = 2, λ = 0.0, ahom = 3.94)
    mesh = refine_uniformly(hypercube(elementtype, n), times = refinements)
    sort_element_nodes!(mesh.elements)
    @show nnodes(mesh)

    # Conductivity per [0, 1]^d bit
    σ = generate_conductivity(mesh, n)

    # Simple lookup to get the conductivity per mesh element
    σ_per_el = conductivity_per_element(mesh, σ)
    interior = list_interior_nodes(mesh)

    A_full = assemble_checkerboard(mesh, σ_per_el, λ)
    Ā_full = assemble_matrix(mesh, (∇u, ∇v) -> ahom * dot(∇u, ∇v))
    b_full = assemble_vector(mesh, identity)
    A = A_full[interior, interior]
    Ā = Ā_full[interior, interior]
    b = b_full[interior]
    x = zeros(nnodes(mesh))
    x̄ = zeros(nnodes(mesh))
    x[interior] .= A \ b
    x̄[interior] .= Ā \ b

    vtk_grid("checkerboard_full", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        vtk_point_data(vtk, x̄, "x_bar")
        vtk_cell_data(vtk, reshape(reinterpret(Float64, σ_per_el), dimension(mesh), :), "σ")
    end

    nothing
end