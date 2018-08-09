using WriteVTK
using Rewrite
using IterativeSolvers

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
            if 1 < yn < m && 1 < xn < n
                stencil[5] += ∂σ
                stencil[i] -= ∂σ
            else
                stencil[5] += 2∂σ
            end
        end

        node = (x - 2) * (n - 2) + (y - 1)

        for (i, (xn, yn)) in enumerate(neighbours)

            # Drop "ghost" cells
            1 < yn < m && 1 < xn < n || continue

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

function save_stuff(N = 4, ξ = (1/√2, 1/√2), ref = 3, Δt = 1.0)

    # Physical width of domain is 2^N, 
    # but we add one cell around it to use σ values for the boundary
    # so basically 2^N + 2h width in total.
    # Grid size h = 2^-ref
    # Number of cells = 2 + 2^(N + ref) along one dimension

    L = 2 ^ N
    n = 2 ^ (N + ref) + 2
    σs = Rewrite.generate_field((n, n), Float64, 8, 20.0, 1.2)
    h = 1 / 2 ^ ref
    xs_boundary = linspace(0, L, n - 1)
    xs_cell = xs_boundary[1:end-1] .+ h / 2
    A, b = finite_volume_div_a_grad(σs, ξ, h)
    x = copy(b)

    vtk = vtk_grid("example", xs_cell, xs_cell)

    for Δt = 2 .^ (N-1)
        scale!(x, 1 / Δt)
        A_l2 = I / Δt + A
        x = A_l2 \ x
        vtk_point_data(vtk, x, "x_$Δt")
        vtk_point_data(vtk, y, "y_$Δt")
    end

    @show extrema(σs) n L 16^(N - 1) extrema(x)

    vtk_point_data(vtk, view(σs, 2:n-1, 2:n-1), "σ")
    vtk_point_data(vtk, b, "b")

    vtk_save(vtk)
end

"""
    poisson_example(N, ref)

Solve -Δu = 2∑xᵢ(2^N - xᵢ) on U = [0, 2^N]^2, u = 0 on ∂U
Exact solution u(x) = ∏xᵢ(2^N - xᵢ).
Maximum = 16^(N - 1)
"""
function poisson_example(N = 4, ref = 3)
    # Physical width of domain is 2^N, 
    # but we add one cell around it to use σ values for the boundary
    # so basically 2^N + 2h width in total.
    # Grid size h = 2^-ref
    # Number of cells = 2 + 2^(N + ref) along one dimension

    L = 2^N
    n = 2^(N + ref) + 2
    σs = ones(n, n)
    h = 1 / 2^ref
    xs_boundary = linspace(0, L, 2^(N + ref) + 1)
    xs_cell = xs_boundary[1:end-1] .+ h/2
    A, _ = finite_volume_div_a_grad(σs, (1.0, 0.0), h)
    b = (2 .* h^2 .* ((xs_cell .* (L .- xs_cell)) .+ (xs_cell .* (L .- xs_cell))'))[:]
    x = A \ b
    @show L 16^(N - 1) - maximum(x)
    vtk = vtk_grid("poisson", xs_cell, xs_cell)
    vtk_point_data(vtk, view(σs, 2:n-1, 2:n-1), "σ")
    vtk_point_data(vtk, x, "x")
    vtk_point_data(vtk, b, "b")
    vtk_save(vtk)
end