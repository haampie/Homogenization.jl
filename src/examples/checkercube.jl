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
    is, js, vs = Vector{Ti}(total), Vector{Ti}(total), Vector{Tv}(total)
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

    # Build the sparse matrix
    return reinterpret(SVector{dim,Tv}, reshape(bs, :))
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
    σ_el = Vector{SVector{dim,Tv}}(nelements(mesh))

    for (idx, el) in enumerate(mesh.elements)
        indices = unsafe_trunc.(Int, mean(get_nodes(mesh, el))).data
        σ_el[idx] = σ.σ[CartesianIndex(indices)]
    end

    σ_el
end

function checkerboard_hypercube_full(n::Int, elementtype::Type{<:ElementType} = Tet{Float64}, refinements = 2, λ = 1.0)
    mesh = refine_uniformly(hypercube(elementtype, n), times = refinements)
    sort_element_nodes!(mesh.elements)
    @show nnodes(mesh)

    # Conductivity per [0, 1]^d bit
    σ = generate_conductivity(mesh, n)

    # Simple lookup to get the conductivity per mesh element
    σ_per_el = conductivity_per_element(mesh, σ)
    interior = list_interior_nodes(mesh)

    return A_full = assemble_checkercube(mesh, σ_per_el, λ)
    Ā_full = assemble_matrix(mesh, (∇u, ∇v) -> 3.0 * dot(∇u, ∇v))
    b_full = assemble_vector(mesh, identity)
    A = A_full[interior, interior]
    b = b_full[interior]

    x = zeros(nnodes(mesh))
    x̄ = zeros(nnodes(mesh))
    x[interior] .= A[interior, interior] \ b[interior]
    x̄[interior] .= Ā[interior, interior] \ b[interior]

    vtk_grid("checkercube_full", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        vtk_point_data(vtk, x̄, "x_bar")
        vtk_cell_data(vtk, reshape(reinterpret(Float64, σ_per_el), dimension(mesh), :), "σ")
    end

    return A_full
end

function checkercube(n::Int, elementtype::Type{<:ElementType} = Tet{Float64}, refinements::Int = 2, save::Int = 1, cycles::Int = 20, steps::Int = 2)
    ### Coarse grid
    base = hypercube(elementtype, n)

    # The correction to the homogenized coefficient (and the correction per step)
    mourrat_σ = 0.0
    Δmourrat_σ = 0.0

    # Unit direction.
    ξ = @SVector rand(dimension(base))
    ξ /= norm(ξ)

    # Conductivity per [0, 1]^d bit
    σ = generate_conductivity(base, n)

    # Simple lookup to get the conductivity per mesh element
    σ_per_el = conductivity_per_element(base, σ)

    ### Coarse grid.
    interior = list_interior_nodes(base)
    F = cholfact(assemble_checkercube(base, σ_per_el, 1.0)[interior,interior])
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
        L2PlusDivAGrad(diff, mass, constraint, 1.0, σ_per_el)
    end

    # Allocate state vectors x, b and r on all levels.
    level_states = map(1 : refinements) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    finest_level = level_states[refinements]

    # Allocate previous v
    v_prev = similar(finest_level.x)

    # Initialize a random x.
    rand!(finest_level.x)
    broadcast_interfaces!(finest_level.x, implicit, refinements)
    apply_constraint!(finest_level.x, refinements, constraint, implicit)

    # Construct a r.h.s.
    ∂ϕ∂xᵢs = partial_derivatives_functionals(refined_mesh(implicit, refinements))
    rhs_aξ∇v!(finest_level.b, ∂ϕ∂xᵢs, implicit, σ_per_el, ξ)

    ωs = [.028,.028,.028,.028,.028,.028,.028,.028] ./ 1.15

    ### Select a handful of 
    center = @SVector fill(0.5 * n + 1, dimension(base))
    radius = float(div(n, 2) - 10)
    subset = select_cells_to_integrate_over(base_mesh(implicit), center, radius)

    ### Save the coarse grid for reference
    # println("Saving")
    # pvd_coarse = paraview_collection("checker_coarse")
    # coarsegridfull = construct_full_grid(implicit, 1)
    # vtk_coarse = vtk_grid("checker_coarse", coarsegridfull)
    # vtk_cell_data(vtk_coarse, reshape(reinterpret(Float64, σ_per_el), dimension(coarsegridfull), :), "σ")
    # selected_guys = zeros(nelements(coarsegridfull))
    # selected_guys[subset] = 1.0
    # vtk_cell_data(vtk_coarse, selected_guys, "Domain of integration")
    # vtk_save(vtk_coarse)
    # collection_add_timestep(pvd_coarse, vtk_coarse, 1.0)

    ## Save a slightly less coarse version of the vₖ's
    # tmpgrid = construct_full_grid(implicit, save)
    # pvd = paraview_collection("checker")
    # vtk_ = vtk_grid("checker_000", tmpgrid)
    # n = nnodes(refined_mesh(implicit, save))
    # bcopy = copy(finest_level.b)
    # broadcast_interfaces!(bcopy, implicit, refinements)
    # vtk_point_data(vtk_, bcopy[1 : n, :], "b")
    # vtk_save(vtk_)
    # collection_add_timestep(pvd, vtk_, 0.0)

    println("Cycling")

    total_mg_steps = 0
    local_sum = zeros(nelements(implicit.base))
    ops = level_operators[refinements]

    for i = 1 : cycles
        vcycle!(implicit, base_level, level_operators, level_states, ωs, refinements, steps)

        # total_mg_steps += 1
        # println("Saving to checker_$(lpad(total_mg_steps,3,0)).vtu")
        # vtk = vtk_grid("checker_$(lpad(total_mg_steps,3,0))", tmpgrid)
        # n = nnodes(refined_mesh(implicit, save))
        # vtk_point_data(vtk, level_states[end].r[1 : n, :], "r")
        # vtk_point_data(vtk, level_states[end].x[1 : n, :], "x")
        # vtk_point_data(vtk, level_states[end].b[1 : n, :], "b")
        # vtk_save(vtk)
        # collection_add_timestep(pvd, vtk, float(total_mg_steps))

        # Compute the correction to the homogenized coefficient

        fill!(local_sum, 0)
        sum_first_term!(local_sum, finest_level.x, ∂ϕ∂xᵢs, implicit, subset, ops, σ_per_el, ξ)
        Δmourrat_σ = sum(local_sum) / area(ops, implicit, subset)
        @show Δmourrat_σ

        # Compute the residual (that lags behind one step!!)
        zero_out_all_but_one!(finest_level.r, implicit, refinements)
        @show vecnorm(finest_level.r)
    end

    mourrat_σ += Δmourrat_σ

    # vtk_save(pvd)

    λ = 1.0

    for mourrat_steps = 2 : 4
        # Our current x becomes our previous x.
        # We keep finest_level.x as initial guess.
        copy!(v_prev, finest_level.x)

        λ /= 2

        # Update the fine grid operator (just λ)
        for operator in level_operators
            operator.λ = λ
        end

        # Construct a new coarse grid operator
        F = cholfact(assemble_checkercube(base, σ_per_el, λ)[interior,interior])
        base_level = BaseLevel(Float64, F, nnodes(base), interior)

        # Update the right-hand side.
        next_rhs!(finest_level.b, finest_level.x, implicit, ops)

        for i = 1 : cycles
            vcycle!(implicit, base_level, level_operators, level_states, ωs, refinements, steps)
            
            # total_mg_steps += 1
            # println("Saving to checker_$(lpad(total_mg_steps,3,0)).vtu")
            # vtk = vtk_grid("checker_$(lpad(total_mg_steps,3,0))", tmpgrid)
            # n = nnodes(refined_mesh(implicit, save))
            # vtk_point_data(vtk, level_states[end].r[1 : n, :], "r")
            # vtk_point_data(vtk, level_states[end].x[1 : n, :], "x")
            # vtk_point_data(vtk, level_states[end].b[1 : n, :], "b")
            # vtk_save(vtk)
            # collection_add_timestep(pvd, vtk, float(total_mg_steps))

            fill!(local_sum, 0)
            sum_terms!(local_sum, finest_level.x, v_prev, implicit, subset, ops)
            Δmourrat_σ = 2^mourrat_steps * sum(local_sum) / area(ops, implicit, subset)
            @show Δmourrat_σ    

            zero_out_all_but_one!(finest_level.r, implicit, refinements)
            @show vecnorm(finest_level.r)
        end

        # Select a new integration domain
        radius = ceil(radius / sqrt(2))
        subset = select_cells_to_integrate_over(base_mesh(implicit), center, radius)

        mourrat_σ += Δmourrat_σ
        @show mourrat_σ

        # println("Saving another coarse grid")
        # vtk_coarse = vtk_grid("checker_coarse_$(lpad(mourrat_steps,3,0))", coarsegridfull)
        # vtk_cell_data(vtk_coarse, reshape(reinterpret(Float64, σ_per_el), dimension(coarsegridfull), :), "σ")
        # selected_guys = zeros(nelements(coarsegridfull))
        # selected_guys[subset] = 1.0
        # vtk_cell_data(vtk_coarse, selected_guys, "Domain of integration")
        # vtk_save(vtk_coarse)
        # collection_add_timestep(pvd_coarse, vtk_coarse, float(mourrat_steps))
    end

    # vtk_save(pvd)
    # vtk_save(pvd_coarse)

    mourrat_σ
end

"""
Returns an ordered list of cell indices that have a centerpoint at most `radius`
away from `center` in the infinity norm.
"""
function select_cells_to_integrate_over(mesh::Mesh{dim,N,Tv,Ti}, center::SVector{dim,Tv}, radius::Tv) where {dim,N,Tv,Ti}
    indices = Vector{Ti}(nelements(mesh))

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
        A_mul_B!(Mv₀_local, ops.mass, v₀_local)

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
        A_mul_B!(Mvₖ_local, ops.mass, vₖ_local)

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