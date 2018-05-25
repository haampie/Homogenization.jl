struct ∫ϕₓᵢϕₓⱼ{dim,num,Tv,Ti}
    ops::SMatrix{dim,dim,SparseMatrixCSC{Tv,Ti},num}
end

"""
    build_local_operators(::MultilevelReference) -> Vector{∫ϕₓᵢϕₓⱼ}

Build the local ϕₓᵢ * ϕₓⱼ operators for each level.
"""
function build_local_operators(ref::MultilevelReference{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    ∫ϕₓᵢϕₓⱼs = Vector{∫ϕₓᵢϕₓⱼ{dim,dim*dim,Tv,Ti}}(length(ref.levels))

    for (i, level) in enumerate(ref.levels)
        ∫ϕₓᵢϕₓⱼs[i] = _build_local_operators(level)
    end

    return ∫ϕₓᵢϕₓⱼs
end

function _build_local_operators(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)

    Nt = nelements(mesh)
    Nn = nnodes(mesh)
    Nq = nquadpoints(quadrature)
    N = 4
        
    # We'll pre-allocate the triples (is, js, vs)
    Is = [Vector{Ti}(N * N * Nt) for i = 1 : 9]
    Js = [Vector{Ti}(N * N * Nt) for i = 1 : 9]
    Vs = [Vector{Tv}(N * N * Nt) for i = 1 : 9]
    
    # The local system matrix
    A_locals = [zeros(N, N) for i = 1 : 9]

    idx = 1

    # Loop over all elements & compute the local system matrix
    @inbounds for element in mesh.elements
        reinit!(element_values, mesh, element)

        # Reset local matrices
        for i = 1 : 9
            fill!(A_locals[i], zero(Tv))
        end

        # Compute
        @inbounds for qp = 1 : Nq, i = 1:N, j = 1:N
            ∇ϕᵢ = get_grad(element_values, i)
            ∇ϕⱼ = get_grad(element_values, j)

            for k = 1 : 3, l = 1 : 3
                A_locals[(k - 1) * 3 + l][i,j] += weights[qp] * ∇ϕᵢ[k] * ∇ϕⱼ[l]
            end
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:N, j = 1:N
            det = get_detjac(element_values)

            for k = 1 : 9
                Is[k][idx] = element[i]
                Js[k][idx] = element[j]
                Vs[k][idx] = A_locals[k][i,j] * det
            end
            idx += 1
        end
    end

    # Build the sparse matrix
    ∫ϕₓᵢϕₓⱼ(SMatrix{3,3,SparseMatrixCSC{Tv,Ti},9}(Tuple(dropzeros!(sparse(Is[i], Js[i], Vs[i], Nn, Nn)) for i = 1 : 9)))
end
