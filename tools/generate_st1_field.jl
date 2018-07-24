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

coord(m, i) = abs(abs(i - m - 1) - m)

function transform!(A::RealFFT{T,3}) where {T}
    # Transform A in the Fourier domain.
    @threads for id = 1 : nthreads()
        _transform!(A, id)
    end

    return nothing
end

function _transform!(A::RealFFT{T,3}, thread_id::Int) where {T}
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
                B[i,j,k] /= (T(1) + √(x² + y² + z²))^1.5
            end
        end
    end
end

function example(n = 10, T = Float64, p = 2)
    FFTW.set_num_threads(p)
    @assert iseven(n)

    # Allocate an n×n×n matrix that is to FFT'd
    @time A = RealFFT(T, n, n, n)

    A_real = to_real(A)
    A_imag = to_complex(A)

    # Fill it with random numbers
    @time A_real .= randn.(T)

    # Apply FFT
    @time A_mul_B!(A_imag, rfft_plan!(A), A_real)

    # Transform things in Fourier space
    @time transform!(A)

    # Apply inverse FFT
    @time A_mul_B!(A_real, irfft_plan!(A), A_imag)

    # Some last transformation
    @time A_real .= exp.(200.0 .* abs.(A_real))

    # Save as cell data
    @time grid = vtk_grid("file", 0:n, 0:n, 0:n)
    @time vtk_cell_data(grid, A_real, "Permeability")
    @time vtk_save(grid)
    
    return nothing
end

