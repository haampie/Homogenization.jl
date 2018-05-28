using Base.LinAlg.axpy!

function A_mul_B!()

"""
Each level state vector is basically a matrix of 
size (# nodes per ref el) × (# number of base elements).
"""
struct LevelState{T,Tv<:AbstractMatrix{T}}
    x::Tv
    b::Tv
    r::Tv
end

"""
We store the local operator and the constraint.
"""
struct LevelOperator{dim,num,Tv,Ti}
    A::∫ϕₓᵢϕₓⱼ{dim,nu,Tv,Ti}
    bc::ZeroDirichletConstraint{Ti}
end

"""
The coarsest grid is the base level. Here we should apply the inverse of `A`.
"""
struct BaseLevel{Tv,Tfact,Tdist<:AbstractMatrix{Tv},Tlocal<:AbstractVector{Tv}}
    A_inv::Tfact
    b::Tdist
    b_base::Tlocal
    x_base::Tlocal
end

"""
Performs a single Richardson iteration
"""
function smooth_step!(implicit::ImplicitFineGrid, ops::LevelOperator, ω, curr::LevelState, level::Int)
    # r ← b
    copy!(curr.r, curr.b)

    # r ← -Ax + r (= b - Ax)
    A_mul_B!(-1.0, implicit.base, ops.A, curr.x, 1.0, curr.r)

    # Accumulate the nodes on the interface and update their local values.
    broadcast_interfaces!(curr.r, implicit, level)

    # Apply the boundary condition.
    apply_constraint!(curr.r, level, ops.bc, implicit::ImplicitFineGrid{dim,N,Tv,Ti})

    # x ← ω * r + x
    axpy!(ω, curr.r, curr.x)
end

function vcycle!(implicit::ImplicitFineGrid, ops::Vector{LevelOperator}, Ps::Vector{SparseMatrixCSC}, levels::Vector{LevelState}, ωs::Vector, k::Int)
    if k == 1
        # Copy lvl[1].r, etc etc.
        throw("Hello world.")
    else
        curr, next = levels[k], levels[k - 1]
        P = Ps[k]

        # Smooth
        for i = 1 : 3
            smooth_step!(implicit, ops[k], ωs[k], curr, k)
        end

        # Restrict: bₖ₋₁ ← Pᵀrₖ
        At_mul_B!(next.b, P, next.r)

        # Cycle: solve PᵀAPxₖ₋₁ = bₖ₋₁ approximately.
        vcycle!(implicit, ops, Ps, levels, ωs, k - 1)

        # Interpolate: xₖ ← Pxₖ₋₁
        A_mul_B!(1.0, P, next.x, 1.0, next.x)

        # Broadcast and boundary conditions.
        broadcast_interfaces!(next.x, implicit, level)
        
        # Smooth
        for i = 1 : 3
            smooth_step!(implicit, ops[k], ωs[k], curr, k)
        end
    end

    return nothing
end