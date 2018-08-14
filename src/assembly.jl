"""
Build a sparse coefficient matrix for a given mesh and bilinear form
"""
function assemble_matrix(mesh::Mesh{dim,N,Tv,Ti}, bf::Function) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)

    Nt = nelements(mesh)
    Nn = nnodes(mesh)
    Nq = nquadpoints(quadrature)

    # We'll pre-allocate the triples (is, js, vs) that are used to
    # construct the sparse matrix A
    is = Vector{Int64}(undef, N * N * Nt)
    js = Vector{Int64}(undef, N * N * Nt)
    vs = Vector{Tv}(undef, N * N * Nt)

    # The local system matrix
    A_local = zeros(N, N)

    idx = 1

    # Loop over all elements & compute the local system matrix
    @inbounds for element in mesh.elements
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))

        # For each quad point
        for qp = 1 : Nq
            w = weights[qp]
            for j = 1:N
                ∇v = get_grad(element_values, j)
                for i = 1:N
                    ∇u = get_grad(element_values, i)
                    A_local[i,j] += w * bf(∇u, ∇v)
                end
            end
        end

        # Copy the local matrix over to the global one
        for i = 1:N, j = 1:N
            iszero(A_local[i,j]) && continue;
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * get_det_jac(element_values)
            idx += 1
        end
    end

    resize!(is, idx - 1)
    resize!(js, idx - 1)
    resize!(vs, idx - 1)

    # Build the sparse matrix
    return sparse(is, js, vs, Nn, Nn)
end

function assemble_matrix_2(mesh::Mesh{dim,N,Tv,Ti}, bf::Function, quad::QuadRule{dim,Nq,Tv} = default_quad(cell_type(mesh))) where {dim,N,Nq,Tv,Ti}
    cell = cell_type(mesh)
    weights = get_weights(quad)
    element_values = init_values(cell, quad, update_gradients | update_det_J)

    Nt = nelements(mesh)
    Nn = nnodes(mesh)

    # We'll pre-allocate the triples (is, js, vs) that are used to
    # construct the sparse matrix A
    is = Vector{Int64}(undef, N * N * Nt)
    js = Vector{Int64}(undef, N * N * Nt)
    vs = Vector{Tv}(undef, N * N * Nt)

    # The local system matrix
    A_local = zeros(N, N)

    idx = 1

    # Loop over all elements & compute the local system matrix
    @fastmath @inbounds for element in mesh.elements
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))

        # For each quad point
        for qp = 1 : Nq
            w = weights[qp]
            for j = 1:N
                ∇v = get_grad(element_values, j)
                for i = 1:N
                    ∇u = get_grad(element_values, i)
                    A_local[i,j] += w * bf(∇u, ∇v)
                end
            end
        end

        # Copy the local matrix over to the global one
        for j = 1:N, i = 1:N
            iszero(A_local[i,j]) && continue;
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * get_det_jac(element_values)
            idx += 1
        end
    end

    resize!(is, idx - 1)
    resize!(js, idx - 1)
    resize!(vs, idx - 1)

    # Build the sparse matrix
    return sparse(is, js, vs, Nn, Nn)
end

"""
Build a vector for a given mesh and functional
"""
function assemble_vector(mesh::Mesh{dim,N,Tv,Ti}, functional::Function) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_det_J)

    Nt = nelements(mesh)
    Nn = nnodes(mesh)
    Nq = nquadpoints(quadrature)

    b = zeros(Nn)
    b_local = zeros(Tv, N)

    # Loop over all elements & compute the local system matrix
    @inbounds for element in mesh.elements
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(b_local, zero(Tv))

        # For each quad point
        @inbounds for qp = 1 : Nq, i = 1:N
            v = get_value(element_values, qp, i)
            b_local[i] += weights[qp] * functional(v)
        end

        # Copy the local vec over to the global one
        @inbounds for i = 1:N
            b[element[i]] += b_local[i] * get_det_jac(element_values)
        end
    end

    return b
end
