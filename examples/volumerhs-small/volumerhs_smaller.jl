using Random
using CUDAdrv
using CUDAnative
using CuArrays
using StaticArrays
using GPUifyLoops # for @unroll macro

function volumerhs!(::Val{N}, rhs, Q, nelem) where {N}
    Nq = N + 1

    s_F = @cuStaticSharedMem eltype(Q) (Nq, Nq)

    r_rhsρ = MArray{Tuple{Nq},eltype(rhs)}(undef)

    e = blockIdx().x
    j = threadIdx().y
    i = threadIdx().x

    @inbounds @unroll for k in 1:Nq
        r_rhsρ[k] = zero(eltype(rhs))
    end

    @inbounds @unroll for k in 1:Nq
        s_F[j, i] = Q[i, j, k, e]

        sync_threads()

        # loop of ξ-grid lines
        @unroll for n = 1:Nq
            r_rhsρ[k] += s_F[j, n]
            r_rhsρ[k] += s_F[i, n]
            r_rhsρ[k] += s_F[n, j]
            r_rhsρ[k] += s_F[n, i]
        end
        sync_threads()
    end

    @inbounds @unroll for k in 1:Nq
        rhs[i, j, k, e] += r_rhsρ[k]
    end

    nothing
end

function main()
    N = 4
    nelem = 4000
    ntrials = 1
    DFloat = Float32

    rnd = MersenneTwister(0)

    Nq = N + 1
    Q = 1 .+ CuArray(rand(rnd, DFloat, Nq, Nq, Nq, nelem))

    rhs = CuArray(zeros(DFloat, Nq, Nq, Nq, nelem))

    @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs!(Val(N), rhs, Q, nelem))

    CUDAdrv.@profile  begin
        @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs!(Val(N), rhs, Q, nelem))
    end

    CUDAnative.@device_code dir="jlir" begin
        @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs!(Val(N), rhs, Q, nelem))
    end

    nothing
end

main()
