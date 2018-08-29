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

function simple_lanczos_example(n = 100, m = 20)
    A = spdiagm(-1 => fill(-1.0, n-1), 0 => fill(2.0, n), 1 => fill(-1.0, n-1))
    x_exact = rand(n)
    b = A * x_exact

    it1 = CGIterable(A + 1.000I, b)

    @time for (i, j) in zip(1:m, it1)

    end

    @time multishift_cg(A, b, m)

    return nothing
end

function multishift_cg(A, b, m)
    n = size(A, 1)

    # Solution vecs
    x₁, x₂, x₃ = zeros(n), zeros(n), zeros(n)

    # Holds the orthonormal basis of the Krylov subspace in standard inner product
    v_prev, v_curr, v_next = zeros(n), zeros(n), zeros(n)

    # Initial basis vector for Krylov subspace
    copyto!(v_curr, b)
    β = norm(b)
    v_curr ./= norm(b)

    # Orthonormal basis for Krylov subspace in inner product induced by A
    # wᵢ is the ith column of V, but then orthogonalized in the (A + λᵢ) inner
    # product via Gram-Schmidt.
    w₁, w₂, w₃ = copy(v_curr), copy(v_curr), copy(v_curr)

    # Shifts
    λ₁, λ₂, λ₃ = 1.00, 0.50, 0.25

    # Tridiagonal matrix ~ Hessenberg matrix. Just store one column.
    t_prev, t_curr, t_next = 0.0, 0.0, 0.0

    # We basically do a root-free Cholesky decomposition of the tridiagonal
    # Hessenberg matrix T, for multiple shifts: T + σI = LΔ⁻¹L' with
    # Δᵢᵢ = Tᵢᵢ - Tᵢ,ᵢ₋₁² / Δᵢ₋₁,ᵢ₋₁
    # Lᵢᵢ = Δᵢᵢ, Lᵢ₋₁,ᵢ = Tᵢ₋₁,ᵢ
    # This means we have a simple two-term recurrence relation for Δᵢᵢ, so we
    # keep track of the current and previous value for all shifts.
    Δ₁_prev, Δ₂_prev, Δ₃_prev = 0.0, 0.0, 0.0
    Δ₁_curr, Δ₂_curr, Δ₃_curr = 0.0, 0.0, 0.0

    # We do not form L, obviously, since it is completely determined by T and
    # Δ. In fact, we only keep track of the last value of the vector that is the
    # solution to the projected problem Ty = e₁‖b‖ = e₁β.
    # To get a residual norm, we need ‖r‖ = |Tₖ₊₁ₖ|inv(T[1:k,1:k])*e₁β,
    # This simplifies to ‖r‖ = |Tₖ₊₁ₖ| * last entry of inv(L')e₁β
    # and the last entry satisfies a simple recurrence as well, so we just
    # call this value yᵢ for problem i. Initially it is β.
    y₁, y₂, y₃ = β, β, β

    @inbounds for k = 1 : m
        # Construct the next basis vector of the Krylov subspace
        mul!(v_next, A, v_curr)

        # Gram-schmidt v_next against the previous vecs
        t_curr = dot(v_curr, v_next)

        if k == 1
            v_next .= v_next .- t_curr .* v_curr
        else
            t_prev = t_next # symmetry of T
            v_next .= v_next .- t_curr .* v_curr .- t_prev .* v_prev
        end

        # Normalize
        t_next = norm(v_next)
        v_next .*= 1 / t_next

        # Update the Cholesky decomp and apply it right to V and left to the rhs
        if k == 1
            Δ₁_curr = t_curr + λ₁
            Δ₂_curr = t_curr + λ₂
            Δ₃_curr = t_curr + λ₃
            y₁ /= Δ₁_curr
            y₂ /= Δ₂_curr
            y₃ /= Δ₃_curr
        else
            Δ₁_curr = t_curr + λ₁ - t_prev * t_prev / Δ₁_prev
            Δ₂_curr = t_curr + λ₂ - t_prev * t_prev / Δ₂_prev
            Δ₃_curr = t_curr + λ₃ - t_prev * t_prev / Δ₃_prev

            y₁ *= -t_prev / Δ₁_curr
            y₂ *= -t_prev / Δ₂_curr
            y₃ *= -t_prev / Δ₃_curr

            w₁ .= v_curr .- w₁ .* (t_prev / Δ₁_prev)
            w₂ .= v_curr .- w₂ .* (t_prev / Δ₂_prev)
            w₃ .= v_curr .- w₃ .* (t_prev / Δ₃_prev)
        end

        # Residual norm for all unknowns
        r₁, r₂, r₃ = abs(t_next * y₁), abs(t_next * y₂), abs(t_next * y₃)

        # TODO: get rid of these copies.
        copyto!(v_prev, v_curr)
        copyto!(v_curr, v_next)

        # Recurrence of the diagonal entries in the T = LΔ⁻¹L' bit.
        Δ₁_prev, Δ₂_prev, Δ₃_prev = Δ₁_curr, Δ₂_curr, Δ₃_curr

        x₁ .+= w₁ .* y₁
        x₂ .+= w₂ .* y₂
        x₃ .+= w₃ .* y₃
    end

    return x₁, x₂, x₃
end
