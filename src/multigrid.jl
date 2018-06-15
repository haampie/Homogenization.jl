using Base.LinAlg.axpy!

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
Construct a new state wrapper.
"""
function LevelState(total_base_elements::Int, total_fine_nodes::Int, Tv::Type{<:Number})
    x = zeros(Tv, total_fine_nodes, total_base_elements)
    b = zeros(x)
    r = zeros(x)
    LevelState{Tv,typeof(x)}(x, b, r)
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
    b = Vector{Tv}(total_nodes)
    b_interior = Vector{Tv}(length(interior_nodes))
    BaseLevel{typeof(F),Tv,Ti,Vector{Tv}}(F, b, b_interior, interior_nodes)
end

"""
Performs a single Richardson iteration
"""
function smoothing_step!(implicit::ImplicitFineGrid, ops::LocalLinearOperator, ω, curr::LevelState, k::Int)
    # r ← b - A * x
    local_residual!(implicit, ops, curr, k)

    # Global residual
    broadcast_interfaces!(curr.r, implicit, k)

    # x ← x + ω * r 
    axpy!(ω, curr.r, curr.x)
end

function vcycle!(implicit::ImplicitFineGrid, base::BaseLevel, ops::Vector{<:LocalLinearOperator}, levels::Vector{<:LevelState}, ωs::Vector, k::Int, steps = 1)
    if k == 1
        broadcast_interfaces!(levels[1].b, implicit, 1)

        # Use the global numbering again.
        copy_to_base!(base.b, levels[1].b, implicit)

        # Copy the interior over.
        copy!(base.b_interior, view(base.b, base.interior_nodes))
        
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
        for i = 1 : steps
            smoothing_step!(implicit, ops[k], ωs[k], curr, k)
        end

        local_residual!(implicit, ops[k], curr, k)

        # Restrict: bₖ₋₁ ← Pᵀrₖ
        restrict_to!(next.b, P, curr.r)
        fill!(next.x, 0.0)

        # Cycle: solve PᵀAPxₖ₋₁ = bₖ₋₁ approximately.
        vcycle!(implicit, base, ops, levels, ωs, k - 1)

        # Interpolate: xₖ ← xₖ + Pxₖ₋₁
        interpolate_and_sum_to!(curr.x, P, next.x)

        # Smooth
        for i = 1 : steps
            smoothing_step!(implicit, ops[k], ωs[k], curr, k)
        end
    end

    return nothing
end