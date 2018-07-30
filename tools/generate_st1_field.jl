using Base: OneTo
using AbstractFFTs: ScaledPlan, normalization
using FFTW
using Base.LinAlg
using Base: tail
using WriteVTK
using Base.Threads: @threads, nthreads

struct RealFFT{T,N,V<:AbstractArray{T,N}}
    data::V
    nx::Int
end

function RealFFT(T::Type, n::Int, ns...)
    @assert iseven(n)
    Is = (n + 2, ns...)
    A = Array{T}(Is...)
    RealFFT{T,length(Is),typeof(A)}(A, n)
end

to_real(A::RealFFT{T,N}) where {T,N} = view(A.data, OneTo(A.nx), ntuple(i -> :, N - 1)...)
to_complex(A::RealFFT{T,N}) where {T,N} = reinterpret(complex(T), A.data, (size(A.data, 1) ÷ 2, tail(size(A.data))...))

rfft_plan!(A::RealFFT{T,N}, flags::Integer=FFTW.ESTIMATE, timelimit::Real=FFTW.NO_TIMELIMIT) where {T,N} =
    FFTW.rFFTWPlan{T,FFTW.FORWARD,true,N}(to_real(A), to_complex(A), 1:N, flags, timelimit)

brfft_plan!(A::RealFFT{T,N}, flags::Integer=FFTW.ESTIMATE, timelimit::Real=FFTW.NO_TIMELIMIT) where {T,N} =
    FFTW.rFFTWPlan{Complex{T},FFTW.BACKWARD,true,N}(to_complex(A), to_real(A), 1:N, flags, timelimit)

irfft_plan!(A::RealFFT{T,N}, args...) where {T,N} =
    ScaledPlan(brfft_plan!(A, args...), normalization(T, size(to_real(A)), 1:N))

@inline coord(m, i) = abs(abs(i - m - 1) - m)

function transform!(A::RealFFT{T,N}, p::T = T(1.5)) where {T,N}
    # Transform A in the Fourier domain.
    for id = 1 : nthreads()
        _transform!(A, id, p)
    end

    return nothing
end

function _transform!(A::RealFFT{T,3}, thread_id::Int, p::T) where {T}
    B = to_complex(A)

    nx, ny, nz = size(B)

    @inbounds for k = thread_id : nthreads() : nz
        z = coord(nz ÷ 2, k)
        z² = T(z * z)
        for j = 1 : ny
            y = coord(ny ÷ 2, j)
            y² = T(y * y)
            for i = 1 : nx
                x = i - 1
                x² = T(x * x)
                B[i,j,k] /= (T(1) + √(x² + y² + z²))^p
            end
        end
    end
end

function _transform!(A::RealFFT{T,2}, thread_id::Int, p::T) where {T}
    B = to_complex(A)

    nx, ny = size(B)

    @inbounds for j = thread_id : nthreads() : ny
        y = coord(ny ÷ 2, j)
        y² = T(y * y)
        for i = 1 : nx
            x = i - 1
            x² = T(x * x)
            B[i,j] /= (T(1) + √(x² + y²))^p
        end
    end
end

function generate_field(ns::NTuple{dim,Int}, T = Float64, threads = 2, α = T(100), p = T(1.5); save = false) where {dim}
    FFTW.set_num_threads(threads)
    @assert 2 ≤ dim ≤ 3
    @assert all(iseven, ns)

    # Allocate an n×n×n matrix that is to FFT'd
    @time A = RealFFT(T, ns...)

    A_real = to_real(A)
    A_imag = to_complex(A)

    # Fill it with random numbers
    @time A_real .= randn.(T)

    # Apply FFT
    @time A_mul_B!(A_imag, rfft_plan!(A), A_real)

    # Transform things in Fourier space
    @time transform!(A, p)

    # Apply inverse FFT
    @time A_mul_B!(A_real, irfft_plan!(A), A_imag)

    # Some last transformation
    @time A_real .= exp.(α .* abs.(A_real))

    # Save as cell data
    if save
        @time grid = vtk_grid("file", ntuple(i -> 0 : ns[i], dim)...)
        @time vtk_cell_data(grid, A_real, "Permeability")
        @time vtk_save(grid)
    end
    
    return A_real
end

function st1_example(n::Int = 128, elt::Type{<:ElementType} = Tri{Float64}, threads = 2; α = 100.0, p = 1.5, λ = 1.0, save = false)
    mesh = hypercube(elt, n)
    σs = generate_field(ntuple(i->n, dimension(mesh)), Float64, threads, α, p, save = save)
    @show extrema(σs)
    interior = list_interior_nodes(mesh)
    @time A = assemble_st1(mesh, σs, λ)
    b = zeros(size(A, 1))
    b[div(n, 2) * n] = 1.0
    # @time b = assemble_vector(mesh, identity)
    x = zeros(b)
    @time x[interior] .= A[interior,interior] \ b[interior]
    
    vtk_grid("st1_example", mesh) do vtk
        vtk_point_data(vtk, x, "x")
    end
end

@propagate_inbounds σ_in_element(mesh::Mesh{dim}, el::NTuple{M}, σ::AbstractArray{Tv,dim}) where {Tv,dim,M} =
    σ[CartesianIndex(unsafe_trunc.(Int, mean(get_nodes(mesh, el))).data)]

"""
Trivial assembly of the st1 matrix in 2d / 3d without refinements in the cells.
Here σs[i] is the conducivity in element i.
"""
function assemble_st1(mesh::Mesh{dim,N,Tv,Ti}, σs::AbstractArray{Tv,dim}, λ::Tv = 1.0) where {dim,N,Tv,Ti}
    cell = cell_type(mesh)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_gradients | update_det_J)
    
    total = N * N * nelements(mesh)
    is, js, vs = Vector{Ti}(total), Vector{Ti}(total), Vector{Tv}(total)
    A_local = zeros(N, N)

    idx = 1
    @inbounds for (e_idx, element) in enumerate(mesh.elements)
        reinit!(element_values, mesh, element)

        # Reset local matrix
        fill!(A_local, zero(Tv))

        σ = σ_in_element(mesh, element, σs)

        # For each quad point
        @inbounds for qp = 1 : nquadpoints(quadrature), i = 1:N, j = 1:N
            u = get_value(element_values, qp, i)
            v = get_value(element_values, qp, j)
            ∇u = get_grad(element_values, i)
            ∇v = get_grad(element_values, j)
            A_local[i,j] += weights[qp] * (λ * u * v + σ * dot(∇u, ∇v))
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:N, j = 1:N
            value = A_local[i,j]
            value == zero(Tv) && continue
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = value * get_det_jac(element_values)
            idx += 1
        end
    end

    resize!(is, idx - 1)
    resize!(js, idx - 1)
    resize!(vs, idx - 1)

    # Build the sparse matrix
    return sparse(is, js, vs, nnodes(mesh), nnodes(mesh))
end