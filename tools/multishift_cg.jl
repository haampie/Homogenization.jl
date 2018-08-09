using SparseArrays, Random, LinearAlgebra

import Base: iterate

struct CGIterable{Tm, Ts, Tv, numT <: Real}
    A::Tm
    x::Ts
    r::Tv
    c::Tv
    u::Tv
    residual::Base.RefValue{numT}
    prev_residual::Base.RefValue{numT}
end

function CGIterable(A, b)
    x = zero(b)
    r = copy(b)
    u = zero(b)
    c = similar(b)

    CGIterable(A, x, r, c, u, Base.Ref(norm(r)), Base.Ref(1.0))
end

function iterate(cg::CGIterable, iteration = 1)
    # u := r + βu (almost an axpy)
    β = cg.residual[]^2 / cg.prev_residual[]^2
    cg.u .= cg.r .+ β .* cg.u

    # c = A * u
    mul!(cg.c, cg.A, cg.u)
    α = cg.residual[]^2 / dot(cg.u, cg.c)

    # Improve solution and residual
    cg.x .+= α .* cg.u
    cg.r .-= α .* cg.c

    cg.prev_residual[] = cg.residual[]
    cg.residual[] = norm(cg.r)

    # Return the residual at item and iteration number as state
    cg.residual[], iteration + 1
end

function multishift_cg(n = 1000)
    A = spdiagm(
        -1 => fill(-1.0, n-1),
         0 => fill(2.0, n), 
         1 => fill(-1.0, n-1)
    )
    
    x_exact = ones(n)
    b = A * x_exact

    x = zero(b)
    r = copy(b)
    u = zero(b)
    c = similar(b)

    it1 = CGIterable(A + 1.000I, b)
    it2 = CGIterable(A + 0.500I, b)
    it3 = CGIterable(A + 0.250I, b)

    for (i, a, b) = zip(1 : 10, it1, it2, it3)
        @show it1.residual[] it2.residual[] it3.residual[]
    end
end

function simple_lanczos(A, b, m = 10)
    n = size(A, 1)
    V = zeros(n, m + 1)
    W = similar(V)
    v₁ = view(V, :, 1)
    copyto!(v₁, b)
    β = norm(b)
    v₁ ./= norm(b)

    T = zeros(m + 1, m)
    ds = zeros(m)
    y = zeros(m)
    y[1] = β

    x = zero(b)

    for k = 1 : m
        vₖ = view(V, :, k)
        vₖ₊₁ = view(V, :, k + 1)
        mul!(vₖ₊₁, A, vₖ)

        T[k,k] = dot(vₖ, vₖ₊₁)

        # Gram-Schmidt either against 1 or 2 vecs.
        if k == 1
            vₖ₊₁ .= vₖ₊₁ .- T[k,k] .* vₖ
        else
            # Exploit symmetry
            T[k-1,k] = T[k,k-1]
            vₖ₊₁ .= vₖ₊₁ .- T[k,k] .* vₖ .- T[k,k-1] .* view(V, :, k-1)
        end

        # Normalize
        T[k+1,k] = norm(vₖ₊₁)
        vₖ₊₁ .*= 1 / T[k+1,k]

        # Update the Cholesky decomp
        if k == 1
            ds[k] = T[k,k]
        else
            ds[k] = T[k,k] - T[k,k-1]^2 / ds[k-1]
        end

        # Update solution thingy
        if k == 1
            y[k] = y[k] / √(ds[k])
        else
            y[k] = -T[k-1,k] / √(ds[k]*ds[k-1]) * y[k-1]
        end

        # Update the W = V * inv(L') term.
        if k == 1
            W[:, k] = view(V, :, k) ./ √(ds[k])
        else
            W[:, k] .= (view(V, :, k) .- view(W, :, k - 1) .* (T[k-1,k] / √(ds[k-1]))) ./ √(ds[k])
        end

        x .+= view(W, :, k) .* y[k]
        @show norm(A * x - b)
    end

    return V, T
end

function simple_lanczos_example(n = 100)
    A = spdiagm(
        -1 => fill(-1.0, n-1),
         0 => fill(2.0, n), 
         1 => fill(-1.0, n-1)
    )
    x_exact = rand(n)
    b = A * x_exact

    it1 = CGIterable(A, b)

    for (a, _) = zip(1:10, it1)
        println(it1.residual[])
    end

    A, simple_lanczos(A, b, 10)...
end