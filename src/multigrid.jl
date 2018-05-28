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
    x = Matrix{Tv}(total_fine_nodes, total_base_elements)
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
Performs a single Richardson iteration
"""
function smoothing_step!(implicit::ImplicitFineGrid, ops::LevelOperator, ω, curr::LevelState, k::Int)
    # r ← b
    copy!(curr.r, curr.b)

    # r ← -Ax + r (= b - Ax)
    A_mul_B!(-1.0, implicit.base, ops.A, curr.x, 1.0, curr.r)

    # Accumulate the nodes on the interface and update their local values.
    broadcast_interfaces!(curr.r, implicit, k)

    # Apply the boundary condition.
    apply_constraint!(curr.r, k, ops.bc, implicit)

    # x ← ω * r + x
    axpy!(ω, curr.r, curr.x)
end

function vcycle!(implicit::ImplicitFineGrid, base::BaseLevel, ops::Vector{<:LevelOperator}, levels::Vector{<:LevelState}, ωs::Vector, k::Int)
    if k == 1
        # Use the global numbering again.
        copy_to_base!(base.b, levels[1].b, implicit)

        # Copy the interior over.
        base.b_interior .= base.b[base.interior_nodes]
        
        # Unfortunately this allocates :s
        tmp = base.A_inv \ base.b_interior

        copy!(base.b_interior, tmp)

        # Apply the boundary condition (should in face go with apply_constraint, but who cares)
        fill!(base.b, 0.0)

        # Copy stuff over.
        base.b[base.interior_nodes] .= base.b_interior

        # Distribute the values to the implicit grid again.
        distribute!(levels[1].x, base.b, implicit)
    else
        curr, next = levels[k], levels[k - 1]
        P = implicit.reference.interops[k - 1]

        # Smooth
        for i = 1 : 5
            smoothing_step!(implicit, ops[k], ωs[k], curr, k)
            @show norm(curr.r)
        end

        # Restrict: bₖ₋₁ ← Pᵀrₖ
        At_mul_B!(next.b, P, curr.r)

        # Cycle: solve PᵀAPxₖ₋₁ = bₖ₋₁ approximately.
        vcycle!(implicit, base, ops, levels, ωs, k - 1)

        # Interpolate: xₖ ← Pxₖ₋₁
        A_mul_B!(1.0, P, next.x, 1.0, curr.x)

        # Broadcast and boundary conditions.
        broadcast_interfaces!(curr.x, implicit, k)
        
        # Smooth
        for i = 1 : 5
            smoothing_step!(implicit, ops[k], ωs[k], curr, k)
            @show norm(curr.r)
        end
    end

    return nothing
end