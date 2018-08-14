using SparseArrays, Random, LinearAlgebra

import Base: iterate

# Standard implementation
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

# Multishift implementation.
function multishift_cg(n = 1000)
    A = spdiagm(
        -1 => fill(-1.0, n-1),
         0 => fill(2.0, n), 
         1 => fill(-1.0, n-1)
    )
    
    x_exact = randn
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

"""
    simple_lanczos(A, b, m = 10)

Solve the problems

(A + σ₁I)x₁ = b
(A + σ₂I)x₂ = b
      ⋮
(A + σₙI)xₙ = b

simultaneously for Hermitian, positive definite A, using the same Krylov
subspace. Assuming σ₁ > … > σₙ, the basic way to look at this is we solve
the problem (A + σₙI)xₙ = b and as a by-product we solve for x₁, …, xₙ₋₁ as
well, reusing the same Krylov subspace.
"""
function simple_lanczos(A, b, m = 10)
    n = size(A, 1)
    V = zeros(n, m + 1)
    W₁ = similar(V)
    W₂ = similar(V)
    W₃ = similar(V)
    v₁ = view(V, :, 1)
    copyto!(v₁, b)
    β = norm(b)
    v₁ ./= norm(b)

    T = zeros(m + 1, m)
    ds = zeros(m, 3)
    y = zeros(m, 3)
    y[1,1] = y[1,2] = y[1,3] = β
    x = zeros(n, 3)

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

        # Update the Cholesky decomp and apply it right to V and left to the rhs
        if k == 1
            ds[k,1] = T[k,k] + 1.00
            ds[k,2] = T[k,k] + 0.50
            ds[k,3] = T[k,k] + 0.25
            y[k,1] /= √(ds[k,1])
            y[k,2] /= √(ds[k,2])
            y[k,3] /= √(ds[k,3])
            W₁[:,k] .= view(V, :, k) ./ √(ds[k,1])
            W₂[:,k] .= view(V, :, k) ./ √(ds[k,2])
            W₃[:,k] .= view(V, :, k) ./ √(ds[k,3])
        else
            ds[k,1] = T[k,k] + 1.00 - T[k,k-1]^2 / ds[k-1,1]
            ds[k,2] = T[k,k] + 0.50 - T[k,k-1]^2 / ds[k-1,2]
            ds[k,3] = T[k,k] + 0.25 - T[k,k-1]^2 / ds[k-1,3]
            y[k,1] = -T[k-1,k] / √(ds[k,1] * ds[k-1,1]) * y[k-1,1]
            y[k,2] = -T[k-1,k] / √(ds[k,2] * ds[k-1,2]) * y[k-1,2]
            y[k,3] = -T[k-1,k] / √(ds[k,3] * ds[k-1,3]) * y[k-1,3]
            W₁[:, k] .= (view(V, :, k) .- view(W₁, :, k-1) .* (T[k-1,k] / √(ds[k-1,1]))) ./ √(ds[k,1])
            W₂[:, k] .= (view(V, :, k) .- view(W₂, :, k-1) .* (T[k-1,k] / √(ds[k-1,2]))) ./ √(ds[k,2])
            W₃[:, k] .= (view(V, :, k) .- view(W₃, :, k-1) .* (T[k-1,k] / √(ds[k-1,3]))) ./ √(ds[k,3])
        end

        x[:, 1] .+= view(W₁, :, k) .* y[k, 1]
        x[:, 2] .+= view(W₂, :, k) .* y[k, 2]
        x[:, 3] .+= view(W₃, :, k) .* y[k, 3]
        @show norm(A * x[:,1] .+ 1.00 .* x[:,1] .- b)
        @show norm(A * x[:,2] .+ 0.50 .* x[:,2] .- b)
        @show norm(A * x[:,3] .+ 0.25 .* x[:,3] .- b)
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

    it1 = CGIterable(A + 1.000I, b)
    it2 = CGIterable(A + 0.500I, b)
    it3 = CGIterable(A + 0.250I, b)

    for (i, _, _) = zip(1 : 20, it1, it2, it3)
        @show it1.residual[] it2.residual[] it3.residual[]
        println()
    end

    A, simple_lanczos(A, b, 20)...
end

function very_simple_multishift_cg_example(n = 100)
    A = spdiagm(
        -1 => fill(-1.0, n-1),
         0 => fill(2.1, n), 
         1 => fill(-1.0, n-1)
    )
    x_exact = rand(n)
    b = A * x_exact

    it1 = CGIterable(A, b)
    it2 = CGIterable(A + 0.500I, b)
    # it3 = CGIterable(A + 0.250I, b)

    for (i, _, _) = zip(1 : 20, it1, it2)
        @show it1.residual[] it2.residual[]
        println()
    end

    very_simple_multishift_cg(A, b, 1.0, 20)
end

function very_simple_multishift_cg(A, b, λ, m)
    n = size(A, 1)
    x₁ = zeros(n)
    x₂ = zeros(n)

    p = copy(b)
    r₁ = copy(b)
    r₂ = copy(b)
    c = similar(b)
    rnrm_prev² = dot(r₁, r₁)
    rnrm_curr² = 1.0

    for i = 1 : m
        mul!(c, A, p)
        dot_p_c = dot(p, c)
        α₁ = rnrm_prev² / dot_p_c
        α₂ = dot(r₁, r₂) / (dot_p_c + λ * dot(p, p))
        @show α₂
        x₁ .+= α₁ .* p
        x₂ .+= α₂ .* p
        r₁ .-= α₁ .* c
        r₂ .= r₂ .- α₂ .* c .- (α₂ * λ) .* p
        rnrm_curr² = dot(r₁, r₁)
        @show √rnrm_curr²
        @show norm(A * x₁ + λ * x₁ - b)
        β = rnrm_curr² / rnrm_prev²
        rnrm_prev² = rnrm_curr²
        p .= r₁ .+ β .* p
    end
end