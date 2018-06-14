function assemble_checkercube(mesh::Tets{Tv,Ti}, σ₁, σ₂, σ₃) where {Tv,Ti}
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
        σ₁, σ₂, σ₃ = σ₁[e_idx], σ₂[e_idx], σ₃[e_idx]

        # For each quad point
        @inbounds for qp = 1 : nquadpoints(quadrature), i = 1:4, j = 1:4
            ∇u = get_grad(element_values, i)
            ∇v = get_grad(element_values, j)
            A_local[i,j] += weights[qp] * (σ₁ * ∇u[1] * ∇v[1] + σ₂ * ∇u[2] * ∇v[2] + σ₃ * ∇u[3] * ∇v[3])
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

function conductivity_per_element(mesh::Tets, n::Int)
    σsgrid₁ = [rand(Bool) ? 1.0 : 9.0 for x = 1 : n, y = 1 : n, z = 1 : n]
    σsgrid₂ = [rand(Bool) ? 1.0 : 9.0 for x = 1 : n, y = 1 : n, z = 1 : n]
    σsgrid₃ = [rand(Bool) ? 1.0 : 9.0 for x = 1 : n, y = 1 : n, z = 1 : n]

    σs₁ = Vector{Float64}(nelements(mesh))
    σs₂ = Vector{Float64}(nelements(mesh))
    σs₃ = Vector{Float64}(nelements(mesh))

    for (idx, )
end

function checkercube(n::Int)
    mesh = cube(n)

    vtk_grid("checkercube", mesh) do vtk
        vtk_cell_data(vtk, σs₁, "σ₁")
        vtk_cell_data(vtk, σs₂, "σ₂")
        vtk_cell_data(vtk, σs₃, "σ₃")
    end

    assemble_checkercube(mesh, σs₁, σs₂, σs₃)


end