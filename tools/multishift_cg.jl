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

    @time for (i, _, _) = zip(1 : m, it1, it2, it3)

    end

    @time lanczos_based_multishift_cg_efficient(A, b, m)

    return nothing
end

###


"""
    lanczos_based_multishift_cg(A, b, m = 10)

Solve the problems

(A + λ₁I)x₁ = b
(A + λ₂I)x₂ = b
(A + λ₃I)x₃ = b

simultaneously for Hermitian, positive definite A, using the same Krylov
subspace. Assuming σ₁ > … > σₙ, the basic way to look at this is we solve
the problem (A + σₙI)xₙ = b and as a by-product we solve for x₁, …, xₙ₋₁ as
well, reusing the same Krylov subspace.
"""
function lanczos_based_multishift_cg(A, b, m)
    n = size(A, 1)

    # Holds the orthonormal basis of the Krylov subspace in standard inner product
    V = zeros(n, m + 1)

    # Orthonormal basis for Krylov subspace in inner product induced by A
    W₁, W₂, W₃ = similar(V), similar(V), similar(V)
    λ₁, λ₂, λ₃ = (1.00, 0.50, 0.25)

    # Initial basis vector for Krylov subspace
    v₁ = view(V, :, 1)
    copyto!(v₁, b)
    β = norm(b)
    v₁ ./= norm(b)

    # Tridiagonal matrix ~ Hessenberg matrix
    T = zeros(m + 1, m)

    # Root-free Cholesky of T for multiple shifts
    # T + σI = LΔ⁻¹L' with Δᵢᵢ = Tᵢᵢ - Tᵢ,ᵢ₋₁² / Δᵢ₋₁,ᵢ₋₁
    # Lᵢᵢ = Δᵢᵢ, Lᵢ₋₁,ᵢ = Tᵢ₋₁,ᵢ.
    Δ = zeros(m, 3)
    y = zeros(m, 3)
    y[1,1] = y[1,2] = y[1,3] = β
    x = zeros(n, 3)

    for k = 1 : m
        # Construct the next basis vector of the Krylov subspace
        vₖ = view(V, :, k)
        vₖ₊₁ = view(V, :, k + 1)
        mul!(vₖ₊₁, A, vₖ)

        # Modified Gram-schmidt: vₖ₊₁ ← (I - VV')vₖ₊₁
        # T[1:k,k] ← V'vₖ₊₁
        # vₖ₊₁ ← vₖ₊₁ / ‖vₖ₊₁‖₂

        T[k,k] = dot(vₖ, vₖ₊₁)

        if k == 1
            vₖ₊₁ .= vₖ₊₁ .- T[k,k] .* vₖ
        else
            # Exploit symmetry
            T[k-1,k] = T[k,k-1]
            vₖ₊₁ .= vₖ₊₁ .- T[k,k] .* vₖ .- T[k,k-1] .* V[:, k-1]
        end

        # Normalize
        T[k+1,k] = norm(vₖ₊₁)
        vₖ₊₁ .*= 1 / T[k+1,k]

        # Update the Cholesky decomp and apply it right to V and left to the rhs
        if k == 1
            Δ[k,1] = T[k,k] + λ₁
            Δ[k,2] = T[k,k] + λ₂
            Δ[k,3] = T[k,k] + λ₃
            y[k,1] /= Δ[k,1]
            y[k,2] /= Δ[k,2]
            y[k,3] /= Δ[k,3]
            W₁[:,k] .= V[:, k]
            W₂[:,k] .= V[:, k]
            W₃[:,k] .= V[:, k]
        else
            Δ[k,1] = T[k,k] + λ₁ - T[k,k-1]^2 / Δ[k-1,1]
            Δ[k,2] = T[k,k] + λ₂ - T[k,k-1]^2 / Δ[k-1,2]
            Δ[k,3] = T[k,k] + λ₃ - T[k,k-1]^2 / Δ[k-1,3]
            y[k,1] = -T[k-1,k] / Δ[k,1] * y[k-1,1]
            y[k,2] = -T[k-1,k] / Δ[k,2] * y[k-1,2]
            y[k,3] = -T[k-1,k] / Δ[k,3] * y[k-1,3]
            W₁[:, k] .= V[:, k] .- W₁[:, k-1] .* (T[k-1,k] / Δ[k-1,1])
            W₂[:, k] .= V[:, k] .- W₂[:, k-1] .* (T[k-1,k] / Δ[k-1,2])
            W₃[:, k] .= V[:, k] .- W₃[:, k-1] .* (T[k-1,k] / Δ[k-1,3])
        end

        r₁ = abs(T[k+1,k] * y[k,1])
        r₂ = abs(T[k+1,k] * y[k,2])
        r₃ = abs(T[k+1,k] * y[k,3])

        x[:, 1] .+= W₁[:, k] .* y[k, 1]
        x[:, 2] .+= W₂[:, k] .* y[k, 2]
        x[:, 3] .+= W₃[:, k] .* y[k, 3]
        @show r₁ r₂ r₃
    end

    return x[:,1], x[:,2], x[:,3]
end


function lanczos_based_multishift_cg_efficient(A, b, m)
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

        # @show r₁ r₂ r₃
    end

    return x₁, x₂, x₃
end
