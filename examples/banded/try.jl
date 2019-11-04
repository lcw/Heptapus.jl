using BandedMatrices, LinearAlgebra, UnicodePlots
using GPUifyLoops, StaticArrays
using Test

using CUDAapi # this will NEVER fail
if has_cuda()
  try
    using CUDAnative, CuArrays
    CuArrays.allowscalar(false)
  catch ex
    # something is wrong with the user's set-up (or there's a bug in CuArrays)
    @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
  end
end

function forward!(b, LU::AbstractArray{T,N}, ::Val{Nq}, ::Val{Nfields},
                  ::Val{Ne_vert}, ::Val{Ne_horz},
                  ::Val{element_bandwidth}) where {T, N, Nq, Nfields,
                                                   Ne_vert, Ne_horz,
                                                   element_bandwidth}
  FT = eltype(b)
  n = Nfields * Nq * Ne_vert
  p = q = element_bandwidth * Nfields * Nq

  l_b = MArray{Tuple{p+1}, FT}(undef)

  @inbounds @loop for h in (1:Ne_horz; blockIdx().x)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for v = 1:element_bandwidth
          @unroll for k = 1:Nq
            @unroll for f = 1:Nfields
              ii  = f + (k - 1) * Nfields + (v - 1) * Nfields * Nq
              l_b[ii] =  Ne_vert ≥ v ? b[i, j, k, f, v, h] : zero(FT)
            end
          end
        end
        l_b[p+1] = Ne_vert > element_bandwidth ?
            b[i, j, 1, 1, element_bandwidth+1, h] : zero(FT)

        for v = 1:Ne_vert
          @unroll for k = 1:Nq
            @unroll for f = 1:Nfields
              jj = f + (k - 1) * Nfields + (v - 1) * Nfields * Nq

              @unroll for ii = 2:p+1
                Lii = N == 2 ? LU[ii+q, jj] : LU[i, j, ii+q, jj, h]
                l_b[ii] -= Lii * l_b[1]
              end

              b[i, j, k, f, v, h] = l_b[1]

              @unroll for ii = 1:p
                l_b[ii] = l_b[ii + 1]
              end

              # idx = jj + p
              # fi = (idx % Nfields) + 1
              # idx2 = idx ÷ Nfields
              # ki = (idx2 % Nq) + 1
              # vi = (idx2 ÷ Nq) + 1

              if jj + p < n
                (idx, fi) = fldmod1(jj + p + 1, Nfields)
                (vi, ki) = fldmod1(idx, Nq)

                l_b[p + 1] = b[i, j, ki, fi, vi, h]
              end
            end
          end
        end
      end
    end
  end

  nothing
end

function backward!(b, LU::AbstractArray{T, N}, ::Val{Nq}, ::Val{Nfields},
                   ::Val{Ne_vert}, ::Val{Ne_horz},
                   ::Val{element_bandwidth}) where {T, N, Nq, Nfields,
                                                    Ne_vert, Ne_horz,
                                                    element_bandwidth}
  FT = eltype(b)
  n = Nfields * Nq * Ne_vert
  q = Nfields * Nq * element_bandwidth

  l_b = MArray{Tuple{q+1}, FT}(undef)

  @inbounds @loop for h in (1:Ne_horz; blockIdx().x)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        @unroll for v = Ne_vert:-1:(Ne_vert - element_bandwidth + 1)
          @unroll for k = Nq:-1:1
            @unroll for f = Nfields:-1:1
              ii  = f + (k - 1) * Nfields + (Ne_vert - v) * Nfields * Nq
              l_b[ii+1] =  b[i, j, k, f, v, h]
            end
          end
        end
        l_b[1] = Ne_vert - element_bandwidth > 0 ?
            b[i, j, Nq, Nfields, Ne_vert - element_bandwidth, h] : zero(FT)


        for v = Ne_vert:-1:1
          @unroll for k = Nq:-1:1
            @unroll for f = Nfields:-1:1
              jj = f + (k - 1) * Nfields + (v - 1) * Nfields * Nq

              l_b[q + 1] /= N == 2 ? LU[q + 1, jj] : LU[i, j, q + 1, jj, h]

              @unroll for ii = 1:q
                Uii = N == 2 ? LU[ii, jj] : LU[i, j, ii, jj, h]
                l_b[ii] -= Uii * l_b[q + 1]
              end

              b[i, j, k, f, v, h] = l_b[q + 1]

              @unroll for ii = q:-1:1
                l_b[ii+1] = l_b[ii]
              end

              if jj - q  > 1
                (idx, fi) = fldmod1(jj - q - 1, Nfields)
                (vi, ki) = fldmod1(idx, Nq)

                l_b[1] = b[i, j, ki, fi, vi, h]
              end
            end
          end
        end
      end
    end
  end

  nothing
end


function band_lu!(A, ::Val{Nq}, ::Val{Nfields}, ::Val{Ne_vert},
                  ::Val{Ne_horz},
                  ::Val{element_bandwidth}) where {Nq, Nfields, Ne_vert,
                                                   Ne_horz, element_bandwidth}
  FT = eltype(A)
  n = Nfields * Nq * Ne_vert
  p = q = Nfields * Nq * element_bandwidth

  @inbounds @loop for h in (1:Ne_horz; blockIdx().x)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        for v = 1:Ne_vert
          for k = 1:Nq
            for f = 1:Nfields
              kk = f + (k - 1) * Nfields + (v - 1) * Nfields * Nq

              Aq = A[i, j, q + 1, kk, h]
              for ii = 1:p
                A[i, j, q + ii + 1, kk, h] /= Aq
              end

              for jj = 1:q
                if jj + kk ≤ n
                  Ajj = A[i, j, q - jj + 1, jj + kk, h]
                  for ii = 1:p
                    A[i, j, q + ii - jj + 1, jj + kk, h] -=
                        A[i, j, q + ii + 1, kk, h] * Ajj
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end

function band_forward!(b, L, p)
  n = size(b, 1)
  for j = 1:n, i = (j+1):min(j+p, n)
    b[i] -= L[i, j]*b[j]
  end
end

function band_backward!(b, U, q)
  n = size(b, 1)
  for j = n:-1:1
    b[j] /= U[j, j]
    for i = max(1,j-q):(j-1)
      b[i] -= U[i, j]*b[j]
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

  T = Float64
  A = T[2 1 0
       1 2 1
       0 1 2]
  F = lu(A, Val(false))

  m = n = size(A, 1)
  p = q = 1

  b = T[4;8;8]

  A\b

  bb = copy(b)
  band_forward!(bb, F.L, p)
  band_backward!(bb, F.U, q)

end

let
  Nq = 2
  Nfields = 2
  Ne_vert = 10
  Ne_horz = 2
  EB = 2

  FT = Float64
  m = n = Nq * Nfields * Ne_vert
  p = q = Nq * Nfields * EB

  A = Array(brand(FT, m, n, p, q)) + 10I
  F = lu(A, Val(false))

  A = extract_banded(A, p, q)
  L = extract_banded(F.L, p, 0)
  U = extract_banded(F.U, 0, q)

  A = repeat(reshape(repeat(A, inner=(Nq*Nq, 1)), Nq, Nq, p+q+1, n), outer=(1,1,1,1,Ne_horz))
  L = repeat(reshape(repeat(L, inner=(Nq*Nq, 1)), Nq, Nq, p+1,   n), outer=(1,1,1,1,Ne_horz))
  U = repeat(reshape(repeat(U, inner=(Nq*Nq, 1)), Nq, Nq, q+1,   n), outer=(1,1,1,1,Ne_horz))

  threads = (Nq, Nq)
  blocks = Ne_horz
  @launch(CPU(), threads=threads, blocks=blocks, band_lu!(A, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz), Val(EB)))

  LL = A[:, :, q+1:end, :, :]
  LL[:, :, 1,:, :] .= one(eltype(LL))
  UU = A[:, :, 1:q+1, :, :]

  @test U ≈ UU
  @test L ≈ LL

end

let

  N = 5
  Nq = N + 1

  # H-S
  # Nfields = 5
  # Ne_vert = 8
  # Ne_horz = 6*6*6

  # Dycoms
  Nfields = 5
  Ne_vert = 60
  Ne_horz = 22*22
  EB = 1

  FT = Float64
  m = n = Nq * Nfields * Ne_vert
  p = q = Nq * Nfields * EB

  A = Array(brand(FT, m, n, p, q)) + 10I
  F = lu(A, Val(false))

  # display(spy(A))
  # display(spy(F.L))
  # display(spy(F.U))

  A = extract_banded(A, p, q)
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

  # d_L = repeat(reshape(repeat(DefaultArray(L), inner=(Nq*Nq, 1)), Nq, Nq, p+1, n), outer=(1,1,1,1,Ne_horz))
  # d_U = repeat(reshape(repeat(DefaultArray(U), inner=(Nq*Nq, 1)), Nq, Nq, q+1, n), outer=(1,1,1,1,Ne_horz))

  d_A = DefaultArray(repeat(reshape(repeat(A, inner=(Nq*Nq, 1)), Nq, Nq, p+q+1, n), outer=(1,1,1,1,Ne_horz)))

  threads = (Nq, Nq)
  blocks = Ne_horz
  @launch(device, threads=threads, blocks=blocks, band_lu!(d_A, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz), Val(EB)))
  d_LU = d_A

  # d_L = d_A[:, :, q+1:end, :, :]
  # d_L[:, :, 1,:, :] .= one(eltype(d_L))
  # d_U = d_A[:, :, 1:q+1, :, :]

  d_b = DefaultArray(b)

  threads = (Nq, Nq)
  blocks = Ne_horz

  LU = Array(d_LU)
  @launch(CPU(), threads=threads, blocks=blocks, forward!(b, LU, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz), Val(EB)))
  @launch(CPU(), threads=threads, blocks=blocks, backward!(b, LU, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz), Val(EB)))

  @test x ≈ Array(b)

  @launch(device, threads=threads, blocks=blocks, forward!(d_b, d_LU, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz), Val(EB)))
  @launch(device, threads=threads, blocks=blocks, backward!(d_b, d_LU, Val(Nq), Val(Nfields), Val(Ne_vert), Val(Ne_horz), Val(EB)))

  @test x ≈ Array(d_b)
end
