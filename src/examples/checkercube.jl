function assemble_checkercube(mesh::Tets{Tv,Ti}, σs::Vector{NTuple{3,Tv}}) where {Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)
    total = 4 * 4 * nelements(mesh)
    is, js, vs = Vector{Ti}(total), Vector{Ti}(total), Vector{Tv}(total)
    A_local = zeros(4, 4)
    idx = 1
    @inbounds for (e_idx, element) in enumerate(mesh.elements)
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))
        σ = σs[e_idx]

        # For each quad point
        @inbounds for qp = 1 : nquadpoints(quadrature), i = 1:4, j = 1:4
            ∇u = get_grad(element_values, i)
            ∇v = get_grad(element_values, j)
            A_local[i,j] += weights[qp] * (σ[1] * ∇u[1] * ∇v[1] + σ[2] * ∇u[2] * ∇v[2] + σ[3] * ∇u[3] * ∇v[3])
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:4, j = 1:4
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * get_det_jac(element_values)
            idx += 1
        end
    end

    # Build the sparse matrix
    return dropzeros!(sparse(is, js, vs, nnodes(mesh), nnodes(mesh)))
end

struct Conductivity{Tv,N}
    n::Int
    σ::Array{NTuple{N,Tv},N}
end

"""
Generate a random (x, y, z) conductance in each unit cube in the domain
of [1, n + 1]^3
"""
generate_conductivity(n::Int) = Conductivity(n, [(rand(Bool) ? 1.0 : 9.0,rand(Bool) ? 1.0 : 9.0,rand(Bool) ? 1.0 : 9.0) for x = 1 : n, y = 1 : n, z = 1 : n])

"""
For convenience this guy will just return a vector `v` s.t. `v[el_idx]` is a
tuple of the conductivity in all spatial directions in that element.
"""
function conductivity_per_element(mesh::Tets, σ::Conductivity)
    σ_el = Vector{NTuple{3,Float64}}(nelements(mesh))

    for (idx, el) in enumerate(mesh.elements)
        x, y, z = unsafe_trunc.(Int, mean(get_nodes(mesh, el)))
        σ_el[idx] = σ.σ[x, y, z]
    end

    σ_el
end

function checkercube(n::Int)
    mesh = refine_uniformly(cube(n), times = 2)
    sort_element_nodes!(mesh.elements)
    σ = generate_conductivity(n)
    σ_per_el = conductivity_per_element(mesh, σ)

    @show nnodes(mesh)

    A = assemble_checkercube(mesh, σ_per_el)
    Ā = assemble_matrix(mesh, (∇u, ∇v) -> 4.3 * dot(∇u, ∇v))
    b = assemble_vector(mesh, identity)
    x = zeros(nnodes(mesh))
    x̄ = zeros(nnodes(mesh))

    interior = list_interior_nodes(mesh)
    x[interior] .= A[interior, interior] \ b[interior]
    x̄[interior] .= Ā[interior, interior] \ b[interior]

    vtk_grid("checkercube", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        vtk_point_data(vtk, x̄, "x_bar")
        vtk_cell_data(vtk, reshape(reinterpret(Int, σ_per_el), 3, :), "σ")
    end
end