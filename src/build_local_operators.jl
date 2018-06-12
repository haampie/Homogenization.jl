import Base.LinAlg: A_mul_B!
using Base.Threads: @threads, nthreads

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

function _build_local_operators(mesh::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)

    Nt = nelements(mesh)
    Nn = nnodes(mesh)
    Nq = nquadpoints(quadrature)
        
    # We'll pre-allocate the triples (is, js, vs)
    Is = [Vector{Ti}(N * N * Nt) for i = 1 : dim * dim]
    Js = [Vector{Ti}(N * N * Nt) for i = 1 : dim * dim]
    Vs = [Vector{Tv}(N * N * Nt) for i = 1 : dim * dim]
    
    # The local system matrix
    A_locals = [zeros(N, N) for i = 1 : dim * dim]

    idx = 1

    # Loop over all elements & compute the local system matrix
    @inbounds for element in mesh.elements
        reinit!(element_values, mesh, element)

        # Reset local matrices
        for i = 1 : dim * dim
            fill!(A_locals[i], zero(Tv))
        end

        # Compute
        @inbounds for qp = 1 : Nq, i = 1:N, j = 1:N
            ∇ϕᵢ = get_grad(element_values, i)
            ∇ϕⱼ = get_grad(element_values, j)

            for k = 1 : dim, l = 1 : dim
                A_locals[(k - 1) * dim + l][i,j] += weights[qp] * ∇ϕᵢ[k] * ∇ϕⱼ[l]
            end
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:N, j = 1:N
            det = get_det_jac(element_values)

            for k = 1 : dim * dim
                Is[k][idx] = element[i]
                Js[k][idx] = element[j]
                Vs[k][idx] = A_locals[k][i,j] * det
            end
            idx += 1
        end
    end

    # Build the sparse matrix
    ∫ϕₓᵢϕₓⱼ(SMatrix{dim,dim,SparseMatrixCSC{Tv,Ti},dim*dim}(Tuple(dropzeros!(sparse(Is[i], Js[i], Vs[i], Nn, Nn)) for i = 1 : dim*dim)))
end

"""
    A_mul_B!(α, ::ImplicitFineGrid, ::∫ϕₓᵢϕₓⱼ, level, x, y)

Compute `y ← α * A * x + y` in a distributed fashion. Note that it does not zero
out `y`. (todo)
"""
function A_mul_B!(α::Tv, base::Mesh{dim,N,Tv,Ti}, ∫ϕₓᵢϕₓⱼ_ops::∫ϕₓᵢϕₓⱼ, x::AbstractMatrix{Tv}, y::AbstractMatrix{Tv}) where {dim,N,Tv,Ti}

    # Circular distribution
    @threads for t = 1 : nthreads()
        do_share_of_mv_product!(t, nthreads(), α, base, ∫ϕₓᵢϕₓⱼ_ops, x, y)
    end
end

"""
The actual mv product that is performed per thread
"""
function do_share_of_mv_product!(thread_id::Int, nthreads::Int, α::Tv, base::Mesh{dim,N,Tv,Ti}, ∫ϕₓᵢϕₓⱼ_ops::∫ϕₓᵢϕₓⱼ, x::AbstractMatrix{Tv}, y::AbstractMatrix{Tv}) where {dim,N,Tv,Ti}

    cell = cell_type(base)
    element_values = ElementValues(cell, default_quad(cell), update_det_J | update_inv_J)

    @inbounds for el_idx = thread_id : nthreads : nelements(base)
        reinit!(element_values, base, base.elements[el_idx])

        Jinv = get_inv_jac(element_values)
        detJ = get_det_jac(element_values)

        P = Jinv' * Jinv

        # Try to avoid making views in a loop here
        offset = (el_idx - 1) * size(x, 1)

        # Apply the ops finally.
        for i = 1 : dim, j = 1 : dim
            scalar = α * P[i, j] * detJ
            scalar == 0 && continue
            my_A_mul_B!(scalar, ∫ϕₓᵢϕₓⱼ_ops.ops[i, j], x, y, offset)
            # A_mul_B!(α * P[i, j] * detJ, ∫ϕₓᵢϕₓⱼ_ops.ops[i, j], view(x, :, el_idx), 1.0, view(y, :, el_idx))
        end
    end
end

function my_A_mul_B!(α, A::SparseMatrixCSC, x::AbstractMatrix, y::AbstractMatrix, offset::Int)
    @inbounds for j = 1 : A.n
        αxj = α * x[j + offset]
        for i = A.colptr[j] : A.colptr[j + 1] - 1
            y[A.rowval[i] + offset] += A.nzval[i] * αxj
        end
    end
    y
end