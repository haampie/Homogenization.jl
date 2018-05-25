"""
Build a sparse coefficient matrix for a given mesh and bilinear form
"""
function assemble_matrix_and_rhs(mesh::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)

    Nt = nelements(mesh)
    Nn = nnodes(mesh)
    Nq = nquadpoints(quadrature)
        
    # We'll pre-allocate the triples (is, js, vs) that are used to
    # construct the sparse matrix A
    is = Vector{Int64}(N * N * Nt)
    js = Vector{Int64}(N * N * Nt)
    vs = Vector{Tv}(N * N * Nt)

    b = zeros(Tv, Nn)
    
    # The local system matrix
    A_local = zeros(N, N)
    b_local = zeros(N)

    idx = 1

    # Loop over all elements & compute the local system matrix
    @inbounds for element in mesh.elements
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))
        fill!(b_local, zero(Tv))

        # For each quad point
        @inbounds for qp = 1 : Nq
            for i = 1:N
                for j = 1:N
                    ∇u = get_grad(element_values, i)
                    ∇v = get_grad(element_values, j)
                    A_local[i,j] += weights[qp] * dot(∇u, ∇v)
                end

                v = get_value(element_values, qp, i)
                b_local[i] += weights[qp] * v
            end
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:N, j = 1:N
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * get_detjac(element_values)
            idx += 1
        end

        # Copy the local vector over to the global one
        @inbounds for i = 1:N
            b[element[i]] += b_local[i] * get_detjac(element_values)
        end
    end

    # Build the sparse matrix
    return dropzeros!(sparse(is, js, vs, Nn, Nn)), b
end
