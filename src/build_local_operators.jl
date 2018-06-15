"""
Any operator that works locally on the implicit fine grid.
"""
abstract type LocalLinearOperator end

struct ∫ϕₓᵢϕₓⱼ{dim,num,Tv,Ti}
    ops::SMatrix{dim,dim,SparseMatrixCSC{Tv,Ti},num}
end

"""
We store the local operator and the constraint.
"""
struct SimpleDiffusion{dim,num,Tv,Ti} <: LocalLinearOperator
    A::∫ϕₓᵢϕₓⱼ{dim,num,Tv,Ti}
    bc::ZeroDirichletConstraint{Ti}
end

"""
Contains all the necessary data to perform a matrix-vector product with
the operator L = I - λ∇⋅σ∇ where σ is constant in each coarse element. Also
contains the Dirichlet boundary constraint.
"""
mutable struct L2PlusDivAGrad{T,U,V,W,X} <: LocalLinearOperator
    diffusion_terms::T
    mass::U
    constraint::V
    λ::W
    σs::X
end

"""
    build_local_diffusion_operators(::MultilevelReference) -> Vector{∫ϕₓᵢϕₓⱼ}

Build the local ϕₓᵢ * ϕₓⱼ operators for each level.
"""
function build_local_diffusion_operators(ref::MultilevelReference{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    ∫ϕₓᵢϕₓⱼs = Vector{∫ϕₓᵢϕₓⱼ{dim,dim*dim,Tv,Ti}}(length(ref.levels))

    for (i, level) in enumerate(ref.levels)
        ∫ϕₓᵢϕₓⱼs[i] = _build_local_diffusion_operators(level)
    end

    return ∫ϕₓᵢϕₓⱼs
end

function build_local_mass_matrices(ref::MultilevelReference{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    Ms = Vector{SparseMatrixCSC{Tv,Ti}}(length(ref.levels))

    for (i, level) in enumerate(ref.levels)
        Ms[i] = mass_matrix(level)
    end

    return Ms
end

function _build_local_diffusion_operators(mesh::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
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

function mass_matrix(mesh::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_det_J)
    total = N * N * nelements(mesh)
    is, js, vs = Vector{Ti}(total), Vector{Ti}(total), Vector{Tv}(total)
    A_local = zeros(N, N)

    idx = 1
    @inbounds for (e_idx, element) in enumerate(mesh.elements)
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))

        # For each quad point
        @inbounds for qp = 1 : nquadpoints(quadrature), i = 1:N, j = 1:N
            u = get_value(element_values, qp, i)
            v = get_value(element_values, qp, j)
            A_local[i,j] += weights[qp] * u * v
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:N, j = 1:N
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * get_det_jac(element_values)
            idx += 1
        end
    end

    # Build the sparse matrix
    return dropzeros!(sparse(is, js, vs, nnodes(mesh), nnodes(mesh)))
end
