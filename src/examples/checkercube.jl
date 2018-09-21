using Statistics: mean

"""
    ahom_for_checkercube(n, type; refinements, tol, max_cycles, k_max, smoothing_steps, boundary_layer, save) → σ_sum, σs, rs

Construct a hypercube [1,n]ᵈ on which a checkerboard pattern is constructed with
unit size length per cell. The dimensionality is defined by the FEM element type that is
begin passed in the second argument (`Tri{Float64}` or `Tet{Float64}`).

Ω = [1,n]ᵈ is the total domain which *includes* the boundary layer as  well, so the user 
should typically provide something like `n = 64 + 2 * 10` and `boundary_layer = 10` to get 
a boundary layer of size `10` and an effective domain of size `64` in each dimension.

A base mesh is constructed with n + 1 nodes in each dimension. This mesh is implicitly 
refined `refinements` times. 

In total we do `k_max` steps of JC's algorithm, where we solve the big-L2-term problem with
multigrid until the tolerance of the homogenized coefficient is met; the `tol` parameter 
is the absolute tolerance on the homogenized coefficient. 

To visualize intermediate solutions as well as the checkerboard pattern, one can use the 
`save` keyword argument. Set `save = 0` to store nothing at all. If `save = 1`, this will 
store the `v`'s on the coarsest grid without refinements. If `save = refinements + 1`, this 
will store the `v`'s on the finest grid -- this will typically make your computer run out of
memory! Best is to set `save = 1` or `save = 2`.

Returns the σ parameter from the theorem which is a correction to the homogenized 
coefficient. In this particular example the homogenized coefficient is ā = 5 - σ.

Also returns some convergence history (intermediate σs and residual norms of the multigrid
step).

Example 2D:
```
# Effective size = 64x64 with boundary layer 84x84.

# With just one refinement we get terrible results!
σ, = ahom_for_checkercube(64 + 2 * 10, Tri{Float64}, 1, 1e-4, 60, 5, 2)
@show σ
σ = 1.6163911040833774

# With two refinements
σ, = ahom_for_checkercube(64 + 2 * 10, Tri{Float64}, 2, 1e-4, 60, 5, 2)
@show σ
σ = 1.8172724552722872

# With three refinements
σ, = ahom_for_checkercube(64 + 2 * 10, Tri{Float64}, 3, 1e-4, 60, 5, 2)
@show σ
σ = 1.9068559447779048
```

Example 3D:
```
σ, = ahom_for_checkercube(20 + 2 * 10, Tet{Float64}, 1, 1e-4, 60, 5, 2)
@show σ
0.7811689150982423

σ, = ahom_for_checkercube(20 + 2 * 10, Tet{Float64}, 2, 1e-4, 60, 5, 2)
@show σ
1.0574764348289638

σ, = ahom_for_checkercube(20 + 2 * 10, Tet{Float64}, 3, 1e-4, 60, 5, 2)
@show σ
1.1930881178271788
```
"""
function ahom_for_checkercube(
    n::Int, 
    elementtype::Type{<:ElementType} = Tet{Float64};
    refinements::Int = 2, 
    tol::AbstractFloat = 1e-4, 
    max_cycles::Int = 60, 
    k_max::Int = 3,
    smoothing_steps::Int = 2,
    boundary_layer::Int = 10,
    save::Int = nothing
)
    0 ≤ save ≤ refinements + 1 || throw(ArgumentError("Parameter `save` can at most be $(refinements+1)"))

    base = hypercube(elementtype, n)

    # This ξ should be an argument to the function, but in the case of the checkerboard it
    # does not really matter anyways
    ξ = @SVector ones(dimension(base))
    ξ /= norm(ξ)

    # Generate conductivity
    σ_per_el = conductivity_per_element(base, generate_conductivity(base, n))

    @info "Building a coarse grid"
    interior = list_interior_nodes(base)
    F = cholesky(assemble_checkercube(base, σ_per_el, 1.0)[interior,interior])
    base_level = BaseLevel(Float64, F, nnodes(base), interior)

    # Maybe store the base grid
    if save != 0
        @info "Saving the grid checkerboard grid"
        vtk_grid("checkerboard", base) do vtk
            # WriteVTK.jl wants us to deliver a matrix of size dim × nelements for 
            # vectorial data, so we just reshape and reinterpret the vector of static 
            # vectors `σ_per_el`.
            vtk_cell_data(vtk, reshape(reinterpret(Float64, σ_per_el), dimension(base), :), "a")
        end
    end

    @info "Building implicit grid"
    # ImplicitFineGrid(base, 1) is just the base grid, so add 1 to actually
    # refine things.
    total_grids = refinements + 1
    implicit = ImplicitFineGrid(base, total_grids)
    nodes, edges, faces = list_boundary_nodes_edges_faces(implicit.base)
    constraint = ZeroDirichletConstraint(nodes, edges, faces)

    @info "Built!" implicit
    @info "Building diffusion operators and mass matrices"

    # Build the local operators.
    diff_terms = build_local_diffusion_operators(implicit.reference)
    mass_terms = build_local_mass_matrices(implicit.reference)
    level_operators = map(zip(diff_terms, mass_terms)) do op
        diff, mass = op
        L2PlusDivAGrad(diff, mass, constraint, 1.0, σ_per_el)
    end

    @info "Allocating state vectors x, b and r"

    # Allocate state vectors x, b and r on all levels.
    level_states = map(1 : total_grids) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    finest_level = level_states[total_grids]

    # Allocate previous v
    v_prev = similar(finest_level.x)

    @info "Initializing a random x with zero b.c."
    rand!(finest_level.x)
    broadcast_interfaces!(finest_level.x, implicit, total_grids)
    apply_constraint!(finest_level.x, total_grids, constraint, implicit)

    @info "Building the initial local r.h.s."
    ∂ϕ∂xᵢs = partial_derivatives_functionals(refined_mesh(implicit, total_grids))
    rhs_aξ∇v!(finest_level.b, ∂ϕ∂xᵢs, implicit, σ_per_el, ξ)

    ωs = [.028,.028,.028,.028,.028,.028,.028,.028] ./ 2.0

    center = @SVector fill(0.5 * n + 1, dimension(base))
    radius = float(div(n, 2) - boundary_layer)
    subset = select_cells_to_integrate_over(base_mesh(implicit), center, radius)

    local_sum = zeros(nelements(implicit.base))
    ops = level_operators[total_grids]
    σs = Vector{Float64}[] # Collect the changes in σ per mg iteration
    rs = Vector{Float64}[] # Collect the residual norms per mg iteration
    λ = 1.0
    σ_sum = 0.0

    for k = 0 : k_max
        @info "Next outer step" k

        # Keep track of increments in σ and residual norms of multigrid
        σs_k = Float64[]
        rs_k = Float64[]

        # Construct a coarse grid operator
        F = cholesky(assemble_checkercube(base, σ_per_el, λ)[interior,interior])
        base_level = BaseLevel(Float64, F, nnodes(base), interior)

        # Solve the next problem
        for i = 1 : max_cycles
            vcycle!(implicit, base_level, level_operators, level_states, ωs, total_grids, smoothing_steps)

            # Initial rhs is special
            fill!(local_sum, 0)
            if k == 0
                sum_first_term!(local_sum, finest_level.x, ∂ϕ∂xᵢs, implicit, subset, ops, σ_per_el, ξ)
            else
                sum_terms!(local_sum, finest_level.x, v_prev, implicit, subset, ops)
            end

            # Compute increment in σ and residual norm
            zero_out_all_but_one!(finest_level.r, implicit, total_grids)
            push!(σs_k, 2^k * sum(local_sum) / area(ops, implicit, subset))
            push!(rs_k, norm(finest_level.r))

            σ_sum′ = σ_sum + last(σs_k)

            @info "Next multigrid step" i last(rs_k) last(σs_k) σ_sum′

            # Check convergence
            if i > 1
                # See if there is still some change in the correction to the homogenized
                # coefficient -- potential issue: we don't really take into account that
                # multigrid might just converge too slowly! But this seems an OK criterion.
                abs(σs_k[end] - σs_k[end-1]) < tol && break
            end
        end

        # Maybe store the intermediate vₖ's.
        if save !== 0
            vtk_grid("ahom_$k", construct_full_grid(implicit, save)) do vtk
                # extract the nodal values of `x` on implicit grid number `save` and 
                # enumerate them as a long vector.
                vtk_point_data(vtk, finest_level.x[1 : nnodes(refined_mesh(implicit, save)), :][:], "v")
            end
        end

        # Our current x becomes our previous x; keep x as initial guess for next round
        copyto!(v_prev, finest_level.x)
        λ /= 2

        # Update the fine grid operator (just λ)
        for operator in level_operators
            operator.λ = λ
        end

        # Update the right-hand side.
        next_rhs!(finest_level.b, finest_level.x, implicit, ops)

        # Select a new integration domain
        radius = ceil(radius / sqrt(2))
        subset = select_cells_to_integrate_over(base_mesh(implicit), center, radius)

        push!(σs, σs_k)
        push!(rs, rs_k)
        σ_sum += last(σs_k)
    end

    σ_sum, σs, rs
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
function assemble_checkercube(mesh::Mesh{dim,N,Tv,Ti}, σs::Vector{SVector{dim,Tv}}, λ::Tv = 1.0) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)

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
function conductivity_per_element(mesh::Mesh{dim}, σ::Conductivity{Tv,dim}) where {Tv,dim}
    σ_el = Vector{SVector{dim,Tv}}(undef, nelements(mesh))

    for (idx, el) in enumerate(mesh.elements)
        indices = unsafe_trunc.(Int, mean(get_nodes(mesh, el))).data
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
    F = cholesky(assemble_checkercube(base, σ_per_el, 0.0)[interior,interior])
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

    ωs = [.28,.060,.060,.060,.060,.060,.060,.028]
    rs = Float64[]

    # Solve the next problem
    for i = 1 : max_cycles
        println("Cycle ", i)
        vcycle!(implicit, base_level, level_operators, level_states, ωs, refinements, 3)

        # Compute increment in σ and residual norm
        zero_out_all_but_one!(finest_level.r, implicit, refinements)
        push!(rs, norm(finest_level.r))
        @show last(rs)
    end

    full_mesh = construct_full_grid(implicit, save)

    vtk_grid("checkercube_full_$refinements", full_mesh) do vtk
        vtk_point_data(vtk, finest_level.x[1 : nnodes(refined_mesh(implicit, save)), :][:], "x")
        # vtk_cell_data(vtk, reshape(reinterpret(Float64, σ_per_el), dimension(full_mesh), :), "σ")
    end

    rs
end

function compare_refinements_on_same_material(refinements = 2 : 7)
    results = []
    for ref = refinements
        push!(results, checkercube(148, Tri{Float64}, ref, 1e-4, 50, 5, 2))
    end
    refinements, results
    #plot(vcat([abs.(results[end][1][i] .- (2results[end][1][i][end] - results[end][1][i][end-1])) for i = 1 : 6]...), yscale = :log10, mark = :o)
    #conv = [sum([results[j][1][i][end] for i = 1 : length(results[j][1])]) for j = 1 : 7]
    #plot([abs.(conv .- (4 * conv[7] - conv[6]) / 3)], yscale = :log10, mark = :x)
end

"""
Returns an ordered list of cell indices that have a centerpoint at most `radius`
away from `center` in the infinity norm.
"""
function select_cells_to_integrate_over(mesh::Mesh{dim,N,Tv,Ti}, center::SVector{dim,Tv}, radius::Tv) where {dim,N,Tv,Ti}
    indices = Vector{Ti}(undef, nelements(mesh))

    idx = 0
    for (i, el) in enumerate(mesh.elements)
        # compute the midpoint
        if maximum(abs.(mean(get_nodes(mesh, el)) - center)) < radius
            indices[idx += 1] = i
        end
    end

    resize!(indices, idx)
end

"""
The initial right-hand side is formed via functional F(v) = -∫aξ⋅∇v. As a is
constant, we just compute ∫∂̂ϕᵢ∂xⱼ on the reference cell, and then use the
pullback to assemble the vector using something like (|J|aξ⋅J⁻¹)∇̂ϕ.

It comes down to summing (linear combination of partial derivatives + M v₀) * v₀
"""
function sum_first_term!(local_sum::Vector{Tv}, v₀::AbstractMatrix{Tv}, ∂ϕ∂xᵢs::Vector{SVector{dim,Tv}}, implicit::ImplicitFineGrid{dim}, subset::Vector{Ti}, ops::L2PlusDivAGrad, σs::Vector{SVector{dim,Tv}}, ξ::SVector{dim,Tv}) where {dim,Tv,Ti}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_inv_J | update_det_J)

    # Avoid `nelements(base)` allocations
    v₀_local = zeros(Tv, nnodes(fine))
    Mv₀_local = similar(v₀_local)

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

        total = zero(Tv)

        # Inner product
        @simd for i = 1 : nnodes(fine)
            total += v₀_local[i] * (dot(∂ϕ∂xᵢs[i], P) + Mv₀_local[i])
        end

        local_sum[idx] = total * get_det_jac(element_values)
    end

    local_sum
end

function sum_terms!(local_sum::Vector{Tv}, vₖ, vₖ₋₁, implicit::ImplicitFineGrid, subset::Vector{Ti}, ops::L2PlusDivAGrad) where {Tv,Ti}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_det_J)

    # Avoid `nelements(base)` allocations
    vₖ_local = zeros(Tv, nnodes(fine))
    Mvₖ_local = similar(vₖ_local)

    @inbounds for idx in subset
        reinit!(element_values, base, base.elements[idx])

        # Copy the thing over
        @simd for i = 1 : nnodes(fine)
            vₖ_local[i] = vₖ[i, idx]
        end

        # Multiply with mass
        mul!(Mvₖ_local, ops.mass, vₖ_local)

        # Inner product
        total = zero(Tv)
        @simd for i = 1 : nnodes(fine)
            total += (vₖ[i, idx] + vₖ₋₁[i, idx]) * Mvₖ_local[i]
        end
        local_sum[idx] += total * get_det_jac(element_values)
    end

    local_sum
end

"""
Compute 1ᵗM1 under selected base cells. Maybe it's better to avoid this
computation and incorporate it in the `sum_terms` and `sum_first_term` functions
"""
function area(ops::L2PlusDivAGrad, implicit::ImplicitFineGrid, subset::Vector{Ti}) where {Ti}
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
of [`ahom_for_checkercube`](@ref).

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

    A_full = assemble_checkercube(mesh, σ_per_el, λ)
    Ā_full = assemble_matrix(mesh, (∇u, ∇v) -> ahom * dot(∇u, ∇v))
    b_full = assemble_vector(mesh, identity)
    A = A_full[interior, interior]
    Ā = Ā_full[interior, interior]
    b = b_full[interior]
    x = zeros(nnodes(mesh))
    x̄ = zeros(nnodes(mesh))
    x[interior] .= A \ b
    x̄[interior] .= Ā \ b

    vtk_grid("checkercube_full", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        vtk_point_data(vtk, x̄, "x_bar")
        vtk_cell_data(vtk, reshape(reinterpret(Float64, σ_per_el), dimension(mesh), :), "σ")
    end

    nothing
end