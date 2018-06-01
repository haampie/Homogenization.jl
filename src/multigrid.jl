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
    b = similar(x)
    r = similar(x)
    LevelState{Tv,typeof(x)}(x, b, r)
end


"""
We store the local operator and the constraint.
"""
struct LevelOperator{dim,num,Tv,Ti}
    A::∫ϕₓᵢϕₓⱼ{dim,num,Tv,Ti}
    bc::ZeroDirichletConstraint{Ti}
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
Compute the residual r = b - A * x locally on each subdomain.
"""
function local_residual!(implicit::ImplicitFineGrid, ops::LevelOperator, curr::LevelState, k::Int)
    # r ← b
    copy!(curr.r, curr.b)

    # r ← r - Ax = b - Ax
    A_mul_B!(-1.0, implicit.base, ops.A, curr.x, curr.r)

    # Apply the boundary condition.
    apply_constraint!(curr.r, k, ops.bc, implicit)
end

"""
Performs a single Richardson iteration
"""
function smoothing_step!(implicit::ImplicitFineGrid, ops::LevelOperator, ω, curr::LevelState, k::Int)
    # r ← b - A * x
    local_residual!(implicit, ops, curr, k)

    # Global residual
    broadcast_interfaces!(curr.r, implicit, k)

    # x ← x + ω * r 
    axpy!(ω, curr.r, curr.x)
end

function vcycle!(implicit::ImplicitFineGrid, base::BaseLevel, ops::Vector{<:LevelOperator}, levels::Vector{<:LevelState}, ωs::Vector, k::Int, debug::Bool = false)
    if k == 1
        broadcast_interfaces!(levels[1].b, implicit, 1)

        # Use the global numbering again.
        copy_to_base!(base.b, levels[1].b, implicit)

        # Copy the interior over.
        base.b_interior .= base.b[base.interior_nodes]
        
        # Unfortunately this allocates :s
        tmp = base.A_inv \ base.b_interior

        # Apply the boundary condition (should in face go with apply_constraint, but who cares)
        fill!(base.b, 0.0)

        # Copy stuff over.
        base.b[base.interior_nodes] .= tmp

        # Distribute the values to the implicit grid again.
        distribute!(levels[1].x, base.b, implicit)
    else
        curr, next = levels[k], levels[k - 1]
        P = implicit.reference.interops[k - 1]

        # Smooth
        debug && println("Level ", k, " now smoothing.")
        for i = 1 : 3
            smoothing_step!(implicit, ops[k], ωs[k], curr, k)
            debug && println("Global residual ≤ ", vecnorm(curr.r))    
        end

        local_residual!(implicit, ops[k], curr, k)
        debug && println("Local residual ≤ ", vecnorm(curr.r))

        # Restrict: bₖ₋₁ ← Pᵀrₖ
        debug && println("Restricting the local residual")
        At_mul_B!(next.b, P, curr.r)
        apply_constraint!(next.b, k - 1, ops[k - 1].bc, implicit)
        # broadcast_interfaces!(next.b, implicit, k - 1)

        # Cycle: solve PᵀAPxₖ₋₁ = bₖ₋₁ approximately.
        debug && println("Going to next level")
        vcycle!(implicit, base, ops, levels, ωs, k - 1, debug)

        # Interpolate: xₖ ← xₖ + Pxₖ₋₁: we basically us rₖ as a temporary.
        debug && println("Interpolating back.")
        A_mul_B!(1.0, P, next.x, 1.0, curr.x)

        # Smooth
        debug && println("Level ", k, ": smoothing.")
        for i = 1 : 3
            smoothing_step!(implicit, ops[k], ωs[k], curr, k)
            debug && println("Global residual ≤ ", vecnorm(curr.r))
        end
    end

    return nothing
end