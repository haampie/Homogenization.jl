using StaticArrays

struct Values{Tvalues,Tgrads,Txs,Tjac,Tinvjac,Tdetjac,Toffset}
    updates::Int
    values::Tvalues
    
    ref_gradients::Tgrads
    gradients::Base.RefValue{Tgrads}
    
    ref_xs::Txs
    xs::Base.RefValue{Txs}
    
    jacobian::Base.RefValue{Tjac}
    inv_jacobian::Base.RefValue{Tinvjac}
    det_jacobian::Base.RefValue{Tdetjac}
    
    offset::Base.RefValue{Toffset}
end

function init_values(cell::Type{<:ElementType{dim,ndof,Tv}}, quad::QuadRule{dim,nquad,Tv}, updates::Int = everything) where {dim,ndof,nquad,Tv}
    # Precompute the values in the basis functions & gradients.
    ϕs = get_basis_funcs(cell)
    xs = get_points(quad)

    # Evaluate the basis functions in the quad points.
    ϕ_values = SVector(ntuple(j -> SVector(ntuple(i -> ϕs[i](xs[j])::Tv, ndof)), nquad))

    # Assume gradients are constant for now.
    ∇ϕ_values_reference = hcat(ntuple(i -> gradient(ϕs[i], @SVector(zeros(Tv,dim))), ndof)...)
    xs_matrix_reference = hcat(xs.data...)

    # Offset
    offset = Base.RefValue(@SVector zeros(Tv, dim))

    return Values(
        updates,
        ϕ_values,
        ∇ϕ_values_reference,
        Base.RefValue(copy(∇ϕ_values_reference)),
        xs_matrix_reference,
        Base.RefValue(copy(xs_matrix_reference)),
        Base.RefValue(@SMatrix zeros(Tv, dim, dim)),
        Base.RefValue(@SMatrix zeros(Tv, dim, dim)),
        Base.RefValue(zero(Tv)),
        offset
    )
end

function reinit!(c::Values, m::Mesh, element)
    if should(c, update_J)
        J, shift = affine_map(m, element)
        c.jacobian[] = J
        c.offset[] = shift
    end

    if should(c, update_inv_J)
        c.inv_jacobian[] = inv(c.jacobian[]')
    end

    if should(c, update_gradients)
        c.gradients[] = c.inv_jacobian[] * c.ref_gradients
    end

    if should(c, update_det_J)
        c.det_jacobian[] = abs(det(c.jacobian[]))
    end

    if should(c, update_x)
        c.xs[] = c.jacobian[] * c.ref_xs .+ c.offset[]
    end
    
    c
end

@inline should(c::Values, i::Int) = c.updates & i == i

"""
Get the transformed gradient of basis function `idx`
"""
@propagate_inbounds get_grad(c::Values, idx::Int) = c.gradients[][:, idx]

"""
Get the values of all basis function in quad point `qp`
"""
@propagate_inbounds get_values(c::Values, qp::Int) = c.values[qp]

"""
Get the values of basis function `idx` in quad poin `qp`
"""
@propagate_inbounds get_value(c::Values, qp::Int, idx::Int) = c.values[qp][idx]

"""
Get inv(J')
"""
@propagate_inbounds get_inv_jac(c::Values) = c.inv_jacobian[]

"""
Get the |J|, the absolute value of the determinant of the affine map
"""
@propagate_inbounds get_det_jac(c::Values) = c.det_jacobian[]

"""
Get the jacobian J.
"""
@propagate_inbounds get_jac(c::Values) = c.jacobian[]

"""
Map a local coord in the ref element to a coord in the actual element
"""
@inline get_x(c::Values, x::SVector) = c.jacobian[] * x + c.offset[]