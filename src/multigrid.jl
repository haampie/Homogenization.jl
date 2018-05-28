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
function smooth_step!(implicit::ImplicitFineGrid, ops::LevelOperator, ω, level::Int)
    # r ← b
    copy!(lvl.r, lvl.b)

    # r ← -Ax + r (= b - Ax)
    A_mul_B!(-1.0, implicit.base, ops.A, lvl.x, 1.0, lvl.r)

    # Accumulate the nodes on the interface and update their local values.
    broadcast_interfaces!(lvl.r, implicit, level)

    # Apply the boundary condition.
    apply_constraint!(lvl.r, level, ops.bc, implicit::ImplicitFineGrid{dim,N,Tv,Ti})

    # x ← ω * r + x
    axpy!(ω, r, x)
end

function vcycle!(As, Ps, lvls, ωs, idx::Int)
    if idx == 1
        # Copy lvl[1].r, etc etc.
        throw("Hello world.")
    else
        lvl, nxt = lvls[idx], lvls[idx - 1]

        # Smooth
        for i = 1 : 3
            smooth_step!(As[idx], lvl, ωs[idx])
        end

        # Restrict
        At_mul_B!(nxt.b, Ps[idx], lvl.r)

        # Cycle
        vcycle!(As, Ps, lvls, ωs, idx - 1)

        # Interpolate
        A_mul_B!(1.0, Ps[idx], nxt.x, 1.0, lvl.x)

        # Broadcast and boundary conditions.
        
        # Smooth
        for i = 1 : 3
            smooth_step!(As[idx], lvl, ωs[idx])
        end
    end

    return nothing
end