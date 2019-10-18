using BandedMatrices, LinearAlgebra, UnicodePlots

using GPUifyLoops
using CUDAnative

using CuArrays
using CuArrays.CUSPARSE
using CuArrays.CUSOLVER

import Adapt
Adapt.adapt_storage(to::Type{<:CuArray}, b::BandedMatrix) = BandedMatrices._BandedMatrix(Adapt.adapt(to, b.data), size(b, 2), b.l, b.u)
Adapt.adapt_storage(to, b::BandedMatrix) = BandedMatrices._BandedMatrix(Adapt.adapt(to, b.data), size(b, 2), b.l, b.u)

using Test

## let
## 
##   m = 15
##   n = 15
##   o = 1000
##   p = 3
##   q = 2
## 
##   A = brand(Float64, m, n, p, q)
##   B = rand(Float64, m, o)
##   X = A\B
##   A * X ≈ B
## 
##   F = lu(A)
## 
##   X = F\B
##   @test A * X ≈ B
## 
##   ## dA = BandedMatrices._BandedMatrix(CuArray(A.data), size(A, 2), A.l, A.u)
##   ## L = CuArray(F.L)
##   ## U = BandedMatrices._BandedMatrix(CuArray(F.U.data), size(F.U, 2), F.U.l, F.U.u)
## 
##   d_A = CuArray(A)
##   d_B = CuArray(B)
## 
##   d_A, d_ipiv = CUSOLVER.getrf!(d_A)
##   d_B         = CUSOLVER.getrs!('N', d_A, d_ipiv, d_B)
## 
## 
##   @test Array(d_B) ≈ A\B
## 
##   ## dA = CuSparseMatrixCSR(sA)
##   ## dB = CuArray(B)
##   ## dX = similar(dB)
##   ## tol = 1e-6
##   ## dX = CUSOLVER.csrlsvlu!(dA,dB,dX,tol,one(Cint),'O')
##   ## dX = CUSOLVER.csrlsvqr!(dA,dB,dX,tol,one(Cint),'O')
## 
## end

function forward!(A, B)
    @inbounds @loop for i in (1:size(A,1);
                              (blockIdx().x-1)*blockDim().x + threadIdx().x)
        A[i] = B[i]
    end
    nothing
end

function backward!(A, B)
    @inbounds @loop for i in (1:size(A,1);
                              (blockIdx().x-1)*blockDim().x + threadIdx().x)
        A[i] = B[i]
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
  FT = Float32

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
  x = similar(b)
  fill!(x, NaN)

  # This is the operation we want to reproduce with a GPUifyLoops kernel
  perm = (4, 3, 5, 1, 2, 6)
  xp = reshape(PermutedDimsArray(x, perm), Nfields*Nq*Ne_vert, Nq*Nq*Ne_horz)
  bp = reshape(PermutedDimsArray(b, perm), Nfields*Nq*Ne_vert, Nq*Nq*Ne_horz)
  xp .= F \ bp

  # threads = 1024
  # blocks = ceil(Int, size(A,1)/threads)
  # @launch(CUDA(), threads=threads, blocks=blocks, kernel!(A, B))

end
