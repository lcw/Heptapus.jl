using BandedMatrices, LinearAlgebra, UnicodePlots
using KernelAbstractions, StaticArrays
using CUDA, CUDAKernels
using KernelAbstractions: get_device
using KernelAbstractions.Extras: @unroll
using Test

@kernel function bandedlu_kernel!(
        LU::AbstractArray,
        ::Val{Nqh},
        ::Val{n},
        ::Val{ku},
        ::Val{kl},
        ::Val{Neh},
    ) where{Nqh, n, ku, kl, Neh}

    eh = @index(Group, Linear)

    ij = @index(Local, Linear)

    @inbounds for v = 1:n
        invUvv = 1/LU[ij, ku + 1, v, eh]

        for p = 1:kl
            LU[ij, ku + 1 + p, v, eh] *= invUvv
        end

        for q = 1:ku
            u = v + q
            if u ≤ n
                Uvu = LU[ij, ku + 1 - q, u, eh]
                for p = 1:kl
                    LU[ij, ku + 1 + p - q, u, eh] -=
                        LU[ij, ku + 1 + p, v, eh] * Uvu
                end
            end
        end
    end
end

@kernel function banded_forward_kernel!(
        x::AbstractArray{T, 3},
        LU::AbstractArray{T, 4},
        b::AbstractArray{T, 3},
        ::Val{Nqh},
        ::Val{n},
        ::Val{ku},
        ::Val{kl},
        ::Val{Neh},
    ) where{Nqh, n, ku, kl, Neh, T}

    # private storage for the part of b we are working on
    p_b = @private T (kl + 1)

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Fill the private storage of b
    @inbounds for v = 1:kl+1
        p_b[v] = v ≤ n ? b[ij, v, eh] : -zero(T)
    end

    # Loop over the columns
    @inbounds for v = 1:n
        # Loop over the rows
        @unroll for p = 1:kl
            # Update element
            Luv = LU[ij, ku + 1 + p, v, eh]
            p_b[p + 1] -= Luv * p_b[1]
        end

        # Pull out the b associated with v
        x[ij, v, eh] = p_b[1]

        # Loop over the rows
        @unroll for p = 1:kl
            # shift the private array back one
            p_b[p] = p_b[p + 1]
        end

        # If we have more elements, get the next value
        if v + kl < n
            p_b[kl + 1] = b[ij, v + kl + 1, eh]
        end
    end
end

@kernel function banded_backward_kernel!(
        x::AbstractArray{T, 3},
        LU::AbstractArray{T, 4},
        b::AbstractArray{T, 3},
        ::Val{Nqh},
        ::Val{n},
        ::Val{ku},
        ::Val{kl},
        ::Val{Neh},
    ) where{Nqh, n, ku, kl, Neh, T}

    # private storage for the part of b we are working on
    p_b = @private T (ku + 1)

    # horizonal element number
    eh = @index(Group, Linear)

    # horizontal degree of freedom
    ij = @index(Local, Linear)

    # Fill the private storage of b
    @inbounds for q = 1:ku + 1
        v = n + 1 - q
        p_b[q] = v > 0 ? b[ij, v, eh] : -zero(T)
    end

    # Loop over the columns
    @inbounds for v = n:-1:1
        # Scale and store the first element of b
        Uvv = LU[ij, ku + 1, v, eh]
        p_b[1] /= Uvv

        # Loop over the rows
        @unroll for q = 1:ku
            # Update element
            Uuv = LU[ij, ku + 1 - q, v, eh]
            p_b[q + 1] -= Uuv * p_b[1]
        end

        x[ij, v, eh] = p_b[1]

        # Loop over the rows
        @unroll for q = 1:ku
            # shift the private array back one
            p_b[q] = p_b[q + 1]
        end

        # If we have more elements, get the next value
        if v - ku > 1
            p_b[ku + 1] = b[ij, v - ku - 1, eh]
        end
    end
end

function extract_banded!(B::AbstractMatrix, A::AbstractMatrix, p, q)
  @assert p ≥ 0
  @assert q ≥ 0

  m, n = size(A)

  for j = 1:n, i = max(1, j - q):min(j + p, n)
    B[q + i - j + 1, j] = A[i, j]
  end

  return B
end

function extract_banded(A::AbstractMatrix, p, q)
  B = similar(A, p + q + 1, size(A, 2))
  fill!(B, 0)
  extract_banded!(B, A, p, q)
end

let
  Nfields = 5
  Nq = (4, 4, 4)

  Nqh = prod(Nq[1:end-1])
  Nqv = Nq[end]

  Neh = 1350
  #Neh = 10
  Nev = 15

  kl = Nqv * Nfields
  ku = Nqv * Nfields

  n = Nfields * Nev * Nqv

  # T, AT = (Float64,  Array)
  T, AT = (Float64, CuArray)

  sA = Array(brand(T, n, n, kl, ku)) + 10I
  sF = lu(sA, Val(false))

  # display(spy(sA))
  # display(spy(sF.L))
  # display(spy(sF.U))

  bA = extract_banded(sA, kl, ku)
  bL = extract_banded(sF.L, kl, 0)
  bU = extract_banded(sF.U, 0, ku)

  A = AT(repeat(reshape(repeat(bA, inner=(Nqh, 1)), Nqh, kl+ku+1, n), outer=(1, 1, 1, Neh)))

  CUDA.@profile begin
      event = Event(get_device(A))
      kernel! = bandedlu_kernel!(get_device(A), (Nqh,))
      event = kernel!(A, Val(Nqh), Val(n), Val(ku), Val(kl), Val(Neh);
                      ndrange = (Nqh * Neh,), dependencies = (event,))
      wait(event)
  end

  L = repeat(reshape(repeat(bL, inner=(Nqh, 1)), Nqh, kl+1,    n), outer=(1, 1, 1, Neh))
  U = repeat(reshape(repeat(bU, inner=(Nqh, 1)), Nqh, ku+1,    n), outer=(1, 1, 1, Neh))
  LL = Array(A[:, ku+1:end, :, :])
  LL[:, 1,:, :] .= one(eltype(LL))
  UU = Array(A[:, 1:ku+1, :, :])
  @test U ≈ UU
  @test L ≈ LL

  bb = rand(T, Nq..., Nfields, Nev, Neh)
  xx = similar(bb)
  perm = (3, 4, 5, 1, 2, 6)
  xp = reshape(PermutedDimsArray(xx, perm), Nfields*Nqv*Nev, Nqh*Neh)
  bp = reshape(PermutedDimsArray(bb, perm), Nfields*Nqv*Nev, Nqh*Neh)
  xp .= sF \ bp

  bb = reshape(bb, Nqh, n, Neh)
  xx = reshape(xx, Nqh, n, Neh)

  b = AT(bb)
  x = similar(b)

  CUDA.@profile begin
      event = Event(get_device(A))
      kernel! = banded_forward_kernel!(get_device(A), (Nqh,))
      event = kernel!(x, A, b, Val(Nqh), Val(n), Val(ku), Val(kl), Val(Neh);
                      ndrange = (Nqh * Neh,), dependencies = (event,))
      kernel! = banded_backward_kernel!(get_device(A), (Nqh,))
      event = kernel!(x, A, x, Val(Nqh), Val(n), Val(ku), Val(kl), Val(Neh);
                      ndrange = (Nqh * Neh,), dependencies = (event,))
      wait(event)
  end

  @test xx ≈ Array(x)
end
