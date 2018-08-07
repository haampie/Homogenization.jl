using WriteVTK
using Rewrite

"""
Given a 2D / 3D grid of scalar conductivity values, we discretize the operator
u->∫-∇⋅σ∇udx. The values of σ are in the center of each finite volume cell, and
we linearly interpolate the σ values on center of the edges. The b.c. is just
u = 0.

    +---+---+---+
 ↑  | . | N | . | <-- We discretize over the domain σ[2:end-1,2:end-1] s.t.
 ↑  +---+---+---+     we always have a north / east / south / west value for
 y  | W | C | E |     σ.
 ↑  +---+---+---+
 ↑  | . | S | . |
    +---+---+---+

      → → x → →
"""
function finite_volume_div_a_grad(σs::AbstractArray{Tv,2}, ξ::NTuple{2,Tv}, h::Tv) where {Tv}

    m, n = size(σs)

    # Remove the boundary values
    cells = (m - 2) * (n - 2)

    # Slightly overestimate the number of unknowns
    is = Vector{Tv}(5 * cells)
    js = Vector{Tv}(5 * cells)
    vs = Vector{Tv}(5 * cells)

    # Rhs
    b = zeros(Tv, cells)
    
    # 5-point stencil
    stencil = zeros(Tv, 5)

    idx = 1

    @inbounds for x = 2 : n - 1, y = 2 : m - 1
        
        neighbours = ((x+1,y), (x-1,y), (x,y+1), (x,y-1))
        fill!(stencil, zero(Tv))

        σ_c = σs[y, x]
        
        for (i, (xn, yn)) in enumerate(neighbours)

            # Boundary σ
            ∂σ = (σ_c + σs[yn, xn]) / 2
            0
            if 1 < xn < n && 1 < yn < m
                stencil[5] += ∂σ
                stencil[i] -= ∂σ
            else
                stencil[5] += 2∂σ
            end
        end

        node = (x - 2) * (n - 2) + (y - 1)

        for (i, (xn, yn)) in enumerate(neighbours)

            # Drop "ghost" cells
            1 < xn < n && 1 < yn < m || continue

            # Store fluxes
            is[idx] = node
            js[idx] = (xn - 2) * (n - 2) + (yn - 1)
            vs[idx] = stencil[i]

            idx += 1
        end

        is[idx] = node
        js[idx] = node
        vs[idx] = stencil[5]
        idx += 1

        b[node] = h/2 * ((σs[y,x+1] - σs[y,x-1]) * ξ[1] + (σs[y+1,x] - σs[y-1,x]) * ξ[2])
    end

    resize!(is, idx - 1)
    resize!(js, idx - 1)
    resize!(vs, idx - 1)

    return sparse(is, js, vs, cells, cells), b
end

function save_stuff(N = 4, ξ = (1/√2, 1/√2), ref = 3)

    # Physical width of domain is 2^N, 
    # but we add one cell around it to use σ values for the boundary
    # so basically 2^N + 2h width in total.
    # Grid size h = 2^-ref
    # Number of cells = 2 + 2^(N + ref) along one dimension
    # We use one additional cell to get a value of σ on the boundary edge
    # 

    n = 2^(N + ref) + 2
    # σs = Rewrite.generate_field((n, n), Float64, 8, 20.0, 1.0)

    σs = linspace(0, 1, n).^2 * ones(n)'

    @show extrema(σs) n

    h = 1 / 2^ref
    xs = linspace(0, 2^N, 2^(N + ref) + 1)
    A, b = finite_volume_div_a_grad(σs, ξ, h)
    λ = 0.01
    # x = (λ*I + A) \ (λ .* b)
    x = A \ b

    vtk = vtk_grid("example", xs, xs)

    vtk_cell_data(vtk, view(σs, 2:n-1, 2:n-1), "σ")
    vtk_cell_data(vtk, x, "x")
    vtk_cell_data(vtk, b, "b")

    vtk_save(vtk)
end