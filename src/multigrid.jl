using LinearAlgebra: axpy!

"""
Each level state vector is basically a matrix of
size (# nodes per ref el) × (# number of base elements).
"""
struct LevelState{T,Tv<:AbstractMatrix{T}}
    x::Tv
    b::Tv
    r::Tv
    p::Tv
    Ap::Tv
end

"""
Construct a new state wrapper.
"""
function LevelState(total_base_elements::Int, total_fine_nodes::Int, Tv::Type{<:Number})
    x = zeros(Tv, total_fine_nodes, total_base_elements)
    b = zero(x)
    r = zero(x)
    p = zero(x)
    Ap = zero(x)
    LevelState{Tv,typeof(x)}(x, b, r, p, Ap)
end

"""
The coarsest grid is the base level. Here we should apply the inverse of `A`.
"""
struct BaseLevel{Tfact,Tv,Ti,Tlocal<:AbstractVector{Tv}}
    A_inv::Tfact
    b::Tlocal
    b_interior::Tlocal
    interior_nodes::Vector{Ti}
end

function BaseLevel(Tv::Type{<:Number}, F, total_nodes::Integer, interior_nodes::Vector{Ti}) where {Ti <: Integer}
    b = Vector{Tv}(undef, total_nodes)
    b_interior = Vector{Tv}(undef, length(interior_nodes))
    BaseLevel{typeof(F),Tv,Ti,Vector{Tv}}(F, b, b_interior, interior_nodes)
end

"""
Performs a few CG iterations
"""
function smoothing_steps!(steps::Integer, implicit::ImplicitFineGrid, ops::LocalLinearOperator, curr::LevelState, k::Int)

    # Global residual
    # r ← b - A * x
    local_residual!(implicit, ops, curr, k)
    broadcast_interfaces!(curr.r, implicit, k)

    copyto!(curr.p, curr.r)
    rsqrprev = dot(curr.r, curr.r) # todo, make sure things are not counted multiple times.

    for i = 1 : steps
        # Global product # Ap = A * p
        fill!(curr.Ap, 0.0)
        mul!(1.0, implicit.base, ops, curr.p, curr.Ap)
        apply_constraint!(curr.Ap, k, ops.constraint, implicit)
        broadcast_interfaces!(curr.Ap, implicit, k)

        # Alpha coefficient
        α = rsqrprev / dot(curr.p, curr.Ap); # todo: dot product
        axpy!(α, curr.p, curr.x)
        axpy!(-α, curr.Ap, curr.r)
        rsqr = dot(curr.r, curr.r) # todo: dot product
        curr.p .= curr.r .+ (rsqr / rsqrprev) .* curr.p # almost an axpy
        rsqrprev = rsqr;
    end
end

function vcycle!(implicit::ImplicitFineGrid, base::BaseLevel, ops::Vector{<:LocalLinearOperator}, levels::Vector{<:LevelState}, k::Int, steps = 2)
    if k == 1
        broadcast_interfaces!(levels[1].b, implicit, 1)

        # Use the global numbering again.
        copy_to_base!(base.b, levels[1].b, implicit)

        # Copy the interior over.
        copyto!(base.b_interior, view(base.b, base.interior_nodes))

        # Unfortunately this allocates :s
        tmp = base.A_inv \ base.b_interior

        # Apply the boundary condition (should maybe go with apply_constraint)
        fill!(base.b, 0.0)

        # Copy stuff over.
        base.b[base.interior_nodes] .= tmp

        # Distribute the values to the implicit grid again.
        distribute!(levels[1].x, base.b, implicit)
    else
        curr = levels[k]
        next = levels[k - 1]
        P = implicit.reference.interops[k - 1]

        # Smooth
        smoothing_steps!(steps, implicit, ops[k], curr, k)

        local_residual!(implicit, ops[k], curr, k)

        # Restrict: bₖ₋₁ ← Pᵀrₖ
        restrict_to!(next.b, P, curr.r)
        fill!(next.x, 0.0)

        # Cycle: solve PᵀAPxₖ₋₁ = bₖ₋₁ approximately.
        vcycle!(implicit, base, ops, levels, k - 1)

        # Interpolate: xₖ ← xₖ + Pxₖ₋₁
        interpolate_and_sum_to!(curr.x, P, next.x)

        # Smooth
        smoothing_steps!(steps, implicit, ops[k], curr, k)
    end

    return nothing
end
