# There's probably a nicer way to do this, but let's just define
# the weighted dot product like this for 2D and 3D.
@propagate_inbounds function weighted_dot(∇u::SVector{3}, σ::SVector{3}, ∇v::SVector{3})
    ∇u[1] * σ[1] * ∇v[1] + ∇u[2] * σ[2] * ∇v[2] + ∇u[3] * σ[3] * ∇v[3]
end

@propagate_inbounds function weighted_dot(∇u::SVector{2}, σ::SVector{2}, ∇v::SVector{2})
    ∇u[1] * σ[1] * ∇v[1] + ∇u[2] * σ[2] * ∇v[2]
end

"""
Build the operator for the bilinear form B[u,v] = ∫uv + λ*a∇u⋅∇v.
"""
function assemble_checkercube(mesh::Mesh{dim,N,Tv,Ti}, σs::Vector{SVector{dim,Tv}}, λ::Tv = 0.0) where {dim,N,Tv,Ti}
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
            A_local[i,j] += weights[qp] * (u * v + λ * weighted_dot(∇u, σ, ∇v))
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

function checkerboard_hypercube_full(n::Int, elementtype::Type{<:ElementType} = Tet{Float64}, refinements = 2)
    mesh = refine_uniformly(hypercube(elementtype, n), times = refinements)
    sort_element_nodes!(mesh.elements)
    @show nnodes(mesh)

    # Conductivity per [0, 1]^d bit
    σ = generate_conductivity(mesh, n)

    # Simple lookup to get the conductivity per mesh element
    σ_per_el = conductivity_per_element(mesh, σ)
    interior = list_interior_nodes(mesh)

    Ā_full = assemble_matrix(mesh, (∇u, ∇v) -> 3.0 * dot(∇u, ∇v))
    A_full = assemble_checkercube(mesh, σ_per_el, 1.0)
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

function checkercube(n::Int, elementtype::Type{<:ElementType} = Tet{Float64}, refinements = 2)
    ### Coarse grid
    base = hypercube(elementtype, n)
    λ = 1.0

    # Conductivity per [0, 1]^d bit
    σ = generate_conductivity(base, n)

    # Simple lookup to get the conductivity per mesh element
    σ_per_el = conductivity_per_element(base, σ)

    interior = list_interior_nodes(base)
    Ac = assemble_checkercube(base, σ_per_el, λ)
    M = mass_matrix(base)
    F = factorize(Ac[interior, interior])
    base_level = BaseLevel(Float64, F, nnodes(base), interior)

    ### Fine grid
    implicit = ImplicitFineGrid(base, refinements)
    nodes, edges, faces = list_boundary_nodes_edges_faces(implicit.base)
    constraint = ZeroDirichletConstraint(nodes, edges, faces)

    # Allocate x, b and r on all levels.
    level_states = map(1 : refinements) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    # Build the local operators.
    diff_terms = build_local_diffusion_operators(implicit.reference)
    mass_terms = build_local_mass_matrices(implicit.reference)
    level_operators = map(zip(diff_terms, mass_terms)) do op
        diff, mass = op
        L2PlusDivAGrad(diff, mass, constraint, 1.0, σ_per_el)
    end
    
    ### Construct a right-hand side - let's keep it constant for now.

    finest_level = level_states[end]
    rand!(finest_level.x)
    broadcast_interfaces!(finest_level.x, implicit, refinements)
    local_rhs!(finest_level.b, implicit)

    ωs = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] ./ n

    tmpgrid = construct_full_grid(implicit, 1)
    pvd = paraview_collection("checker")

    println("Cycling")
    for i = 1 : 20
        vcycle!(implicit, base_level, level_operators, level_states, ωs, refinements, 4)
        zero_out_all_but_one!(finest_level.r, implicit, refinements)
        println(vecnorm(finest_level.r))

        println("Saving to checker_$(lpad(i,3,0)).vtu")
        vtk = vtk_grid("checker_$(lpad(i,3,0))", tmpgrid)
        n = nnodes(refined_mesh(implicit, 1))
        vtk_point_data(vtk, level_states[end].r[1 : n, :], "r")
        vtk_point_data(vtk, level_states[end].x[1 : n, :], "x")
        vtk_save(vtk)
        collection_add_timestep(pvd, vtk, float(i))
    end
    vtk_save(pvd)
end