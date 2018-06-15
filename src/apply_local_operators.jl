import Base.LinAlg: A_mul_B!
using Base.Threads: @threads, nthreads

"""
Compute the residual r = b - A * x locally on each subdomain.
"""
function local_residual!(implicit::ImplicitFineGrid, A::SimpleDiffusion, curr::LevelState, k::Int)
    # r ← b
    copy!(curr.r, curr.b)

    # r ← r - Ax = b - Ax
    A_mul_B!(-1.0, implicit.base, A, curr.x, curr.r)

    # Apply the boundary condition.
    apply_constraint!(curr.r, k, A.bc, implicit)
end

function local_residual!(implicit::ImplicitFineGrid, A::L2PlusDivAGrad, curr::LevelState, k::Int)
    # r ← b
    copy!(curr.r, curr.b)

    # r ← r - Ax = b - Ax
    A_mul_B!(-1.0, implicit.base, A, curr.x, curr.r)

    # Apply the boundary condition.
    apply_constraint!(curr.r, k, A.constraint, implicit)
end




### SimpleDiffusion MV-product

"""
    A_mul_B!(α, ::ImplicitFineGrid, ::SimpleDiffusion, level, x, y)

Compute `y ← α * A * x + y` in a distributed fashion. Note that it does not zero
out `y`. (todo)
"""
function A_mul_B!(α::Tv, base::Mesh{dim,N,Tv,Ti}, A::SimpleDiffusion, x::AbstractMatrix{Tv}, y::AbstractMatrix{Tv}) where {dim,N,Tv,Ti}

    # Circular distribution
    @threads for t = 1 : nthreads()
        do_share_of_mv_product!(t, nthreads(), α, base, A, x, y)
    end
end

"""
The actual mv product that is performed per thread
"""
function do_share_of_mv_product!(thread_id::Int, nthreads::Int, α::Tv, base::Mesh{dim,N,Tv,Ti}, A::SimpleDiffusion, x::AbstractMatrix{Tv}, y::AbstractMatrix{Tv}) where {dim,N,Tv,Ti}

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
            my_A_mul_B!(α * P[i, j] * detJ, A.A[i, j], x, y, offset)
        end
    end
end




### L2PlusDivAGrad MV-product

"""
    A_mul_B!(α, ::ImplicitFineGrid, ::L2PlusDivAGrad, level, x, y)

Compute `y ← α * A * x + y` in a distributed fashion. Note that it does not zero
out `y`. (todo)
"""
function A_mul_B!(α::Tv, base::Mesh{dim,N,Tv,Ti}, A::L2PlusDivAGrad, x::AbstractMatrix{Tv}, y::AbstractMatrix{Tv}) where {dim,N,Tv,Ti}

    # Circular distribution
    # @threads for t = 1 : nthreads()
        do_share_of_mv_product!(1, 1, α, base, A, x, y)
    # end
end

function do_share_of_mv_product!(thread_id::Int, nthreads::Int, α::Tv, base::Mesh{dim,N,Tv,Ti}, A::L2PlusDivAGrad, x::AbstractMatrix{Tv}, y::AbstractMatrix{Tv}) where {dim,N,Tv,Ti}

    cell = cell_type(base)
    element_values = ElementValues(cell, default_quad(cell), update_det_J | update_inv_J)

    @inbounds for el_idx = thread_id : nthreads : nelements(base)
        reinit!(element_values, base, base.elements[el_idx])

        Jinv = get_inv_jac(element_values)
        detJ = get_det_jac(element_values)

        # Here we incorporate the σ parameter
        P = Jinv' * (A.σs[el_idx] .* Jinv)

        # Try to avoid making views in a loop here
        offset = (el_idx - 1) * size(x, 1)

        # We're multiplying with the λ∇⋅σ∇ term first
        scalar = α * A.λ * detJ

        # Diffusion part
        for i = 1 : dim, j = 1 : dim
            my_A_mul_B!(scalar * P[i, j], A.diffusion_terms.ops[i, j], x, y, offset)
        end

        # Mass matrix term
        my_A_mul_B!(α * detJ, A.mass, x, y, offset)
    end
end

"""
A version of y ← α * A * x + y that does not make views when operating on a column.
"""
function my_A_mul_B!(α, A::SparseMatrixCSC, x::AbstractMatrix, y::AbstractMatrix, offset::Int)
    @inbounds for j = 1 : A.n
        αxj = α * x[j + offset]
        for i = A.colptr[j] : A.colptr[j + 1] - 1
            y[A.rowval[i] + offset] += A.nzval[i] * αxj
        end
    end
    y
end