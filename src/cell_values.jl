using ForwardDiff.gradient
using Base: @propagate_inbounds

abstract type QuadRule{dim,nquad,Tv} end

abstract type TetQuad4{Tv} <: QuadRule{3,4,Tv} end
abstract type TriQuad3{Tv} <: QuadRule{2,3,Tv} end

function get_points(::Type{TetQuad4{Tv}}) where {Tv}
    a, b = (Tv(5) + Tv(3) * √Tv(5)) / Tv(20), (Tv(5) - √Tv(5)) / Tv(20)

    return SVector((
        SVector{3,Tv}(a,b,b),
        SVector{3,Tv}(b,a,b),
        SVector{3,Tv}(b,b,a),
        SVector{3,Tv}(b,b,b)
    ))
end

function get_weights(::Type{TetQuad4{Tv}}) where {Tv}
    w = Tv(1) / Tv(24)
    return SVector((w, w, w, w))
end

function get_points(::Type{TriQuad3{Tv}}) where {Tv}
    return SVector((
        SVector{2,Tv}(0.0, 0.5),
        SVector{2,Tv}(0.5, 0.0),
        SVector{2,Tv}(0.5, 0.5),
    ))
end

function get_weights(::Type{TriQuad3{Tv}}) where {Tv}
    w = Tv(1/6)
    return SVector((w, w, w, w))
end

@inline default_quad(::Type{Tri{Tv}}) where {Tv} = TriQuad3{Tv}
@inline default_quad(::Type{Tet{Tv}}) where {Tv} = TetQuad4{Tv}
@inline nquadpoints(::Type{<:QuadRule{dim,nquad}}) where {dim,nquad} = nquad

get_basis_funcs(::Type{Tri{Tv}}) where {Tv} = (
    x -> Tv(1) - x[1] - x[2],
    x -> x[1],
    x -> x[2]
)

get_basis_funcs(::Type{Tet{Tv}}) where {Tv} = (
    x -> Tv(1) - x[1] - x[2] - x[3],
    x -> x[1],
    x -> x[2],
    x -> x[3]
)

mutable struct Wrap{T}
    value::T
end

struct ElementValues{dim,nquad,nfuncs,Tv,S,U}
    updates::Int
    values::SVector{nquad,SVector{nfuncs,Tv}}
    gradients::MMatrix{dim,nquad,Tv,S}
    ref_gradients::SMatrix{dim,nquad,Tv,S}
    jacobian::MMatrix{dim,dim,Tv,U}
    inv_jacobian::MMatrix{dim,dim,Tv,U}
    det_jacobian::Wrap{Tv}
    xs::MMatrix{dim,nquad,Tv,S}
    ref_xs::SMatrix{dim,nquad,Tv,S}
end

const update_J         = 1 << 0
const update_inv_J     = (1 << 1) | update_J
const update_gradients = (1 << 2) | update_inv_J | update_J
const update_det_J     = (1 << 3) | update_J
const update_x         = 1 << 4
const everything = update_gradients | update_J | update_det_J | update_inv_J | update_x

function ElementValues(cell::Type{<:ElementType{dim,ndof,Tv}}, quad::Type{<:QuadRule{dim,nquad,Tv}}, updates::Int = everything) where {dim,ndof,nquad,Tv}
    # Precompute the values in the basis functions & gradients.
    ϕs = get_basis_funcs(cell)
    xs = get_points(quad)

    # Evaluate the basis functions in the quad points.
    ϕ_values = SVector{nquad,SVector{ndof,Tv}}(Tuple(SVector(Tuple(ϕ(x)::Tv for ϕ in ϕs)) for x in xs))

    # Assume gradients are constant for now.
    ∇ϕ_values = hcat([gradient(ϕs[i], @SVector(zeros(Tv,dim))) for i = 1 : length(ϕs)]...)

    xs_matrix = hcat(xs...)

    return ElementValues(
        updates,
        ϕ_values,
        @MMatrix(zeros(dim,nquad)),
        ∇ϕ_values,
        @MMatrix(zeros(dim,dim)),
        @MMatrix(zeros(dim,dim)),
        Wrap(zero(Tv)),
        @MMatrix(zeros(dim,nquad)),
        xs_matrix,
    )
end

@propagate_inbounds function reinit!(c::ElementValues, m::Mesh, element)

    if should(c, update_J)
        J, shift = affine_map(m, element) 
        copy!(c.jacobian, J)
    end

    if should(c, update_inv_J)
        copy!(c.inv_jacobian, inv(c.jacobian'))
    end

    if should(c, update_gradients)
        A_mul_B!(c.gradients, c.inv_jacobian, c.ref_gradients)
    end

    if should(c, update_det_J)
        c.det_jacobian.value = abs(det(c.jacobian))
    end

    if should(c, update_x)
        copy!(c.xs, c.jacobian * c.xs .+ affine_map_shift(m, element))
    end
end

@inline should(c::ElementValues, i::Int) = c.updates & i == i

"""
Get the transformed gradient of basis function `idx`
"""
@propagate_inbounds get_grad(c::ElementValues, idx::Int) = c.gradients[:, idx]

"""
Get the values of all basis function in quad point `qp`
"""
@propagate_inbounds get_values(c::ElementValues, qp::Int) = c.values[qp]

"""
Get the values of basis function `idx` in quad poin `qp`
"""
@propagate_inbounds get_value(c::ElementValues, qp::Int, idx::Int) = c.values[qp][idx]

"""
Get inv(J')
"""
@propagate_inbounds get_inv_jac(c::ElementValues) = c.inv_jacobian

"""
Get the |J|, the absolute value of the determinant of the affine map
"""
@propagate_inbounds get_det_jac(c::ElementValues) = c.det_jacobian.value