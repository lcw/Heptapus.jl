using BandedMatrices, LinearAlgebra, UnicodePlots
using GPUifyLoops
using Test

using CUDAapi # this will NEVER fail
if has_cuda()
  try
    using CUDAnative, CuArrays
  catch ex
    # something is wrong with the user's set-up (or there's a bug in CuArrays)
    @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
  end
end

function forward!(b, L, ::Val{Nq}, ::Val{Nfields}, ::Val{Ne_vert}, ::Val{Ne_horz}) where {Nq, Nfields, Ne_vert, Ne_horz}
  n = Nfields * Nq * Ne_vert
  p = Nfields * Nq
  q = 0

  @loop for h in (1:Ne_horz; blockIdx().x)
    @loop for i in (1:Nq; threadIdx().x)
      @loop for j in (1:Nq; threadIdx().y)
        for v = 1:Ne_vert
          for k = 1:Nq
            for f = 1:Nfields
              jj = f + (k - 1) * Nfields + (v - 1) * Nfields * Nq

              for ii = (jj + 1):min(jj + p, n)
                (idx, fi) = fldmod1(ii, Nfields)
                (vi, ki) = fldmod1(idx, Nq)

                b[i, j, ki, fi, vi, h] -= L[q + ii - jj + 1, jj] * b[i, j, k, f, v, h]
              end
            end
          end
        end
      end
    end
  end

  nothing
end

function backward!(b, U, ::Val{Nq}, ::Val{Nfields}, ::Val{Ne_vert}, ::Val{Ne_horz}) where {Nq, Nfields, Ne_vert, Ne_horz}
  n = Nfields * Nq * Ne_vert
  q = Nfields * Nq

  @loop for h in (1:Ne_horz; blockIdx().x)
    @loop for i in (1:Nq; threadIdx().x)
      @loop for j in (1:Nq; threadIdx().y)
        for v = 1:Ne_vert
          for k = 1:Nq
            for f = 1:Nfields
              jj = f + (k - 1) * Nfields + (v - 1) * Nfields * Nq

              b[i, j, k, f, v, h] /= U[q + 1, jj]

              for ii = max(1, jj - q):(jj - 1)
                (idx, fi) = fldmod1(ii, Nfields)
                (vi, ki) = fldmod1(idx, Nq)

                b[i, j, ki, fi, vi, h] -= U[q + ii - jj + 1, jj] * b[i, j, k, f, v, h]
              end
            end
          end
        end
      end
    end
  end

  nothing
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
  extract_banded!(similar(A, p + q + 1, size(A, 2)), A, p, q)
end

let

  N = 5
  Nq = N + 1
  Nfields = 5
  Ne_vert = 8
  Ne_horz = 6*6*6
  FT = Float64

  m = n = Nq * Nfields * Ne_vert
  p = q = Nq * Nfields

  A = Array(brand(FT, m, n, p, q))
  F = lu(A, Val(false))

  # display(spy(A))
  # display(spy(F.L))
  # display(spy(F.U))

  L = extract_banded(F.L, p, 0)
  U = extract_banded(F.U, 0, q)

  b = rand(FT, Nq, Nq, Nq, Nfields, Ne_vert, Ne_horz)
  borig = copy(b)
  x = similar(b)
  fill!(x, NaN)

  # This is the operation we want to reproduce with a GPUifyLoops kernel
  perm = (4, 3, 5, 1, 2, 6)
  xp = reshape(PermutedDimsArray(x, perm), Nfields*Nq*Ne_vert, Nq*Nq*Ne_horz)
  bp = reshape(PermutedDimsArray(b, perm), Nfields*Nq*Ne_vert, Nq*Nq*Ne_horz)
  xp .= F \ bp

  # Move data to the GPU
  DefaultArray = has_cuda() ? CuArray : Array
  device = has_cuda() ? CUDA() : CPU()

  d_L = DefaultArray(L)
  d_U = DefaultArray(U)
  d_b = DefaultArray(b)

  threads = (Nq, Nq)
  blocks = Ne_horz
  @launch(CPU(), threads=threads, blocks=blocks, forward!(b, L, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz)))
  @launch(CPU(), threads=threads, blocks=blocks, backward!(b, U, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz)))

  @launch(device, threads=threads, blocks=blocks, forward!(d_b, d_L, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz)))
  @launch(device, threads=threads, blocks=blocks, backward!(d_b, d_U, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz)))

  x ≈ Array(d_b)
end
