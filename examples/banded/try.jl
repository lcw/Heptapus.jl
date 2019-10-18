using BandedMatrices, LinearAlgebra, UnicodePlots

using CuArrays
using CuArrays.CUSPARSE
using CuArrays.CUSOLVER

using Test

let

  m = 15
  n = 15
  o = 1000
  p = 3
  q = 2

  A = brand(Float64, m, n, p, q)
  B = rand(Float64, m, o)
  X = A\B
  A * X ≈ B

  F = lu(A)

  X = F\B
  @test A * X ≈ B

  ## dA = BandedMatrices._BandedMatrix(CuArray(A.data), size(A, 2), A.l, A.u)
  ## L = CuArray(F.L)
  ## U = BandedMatrices._BandedMatrix(CuArray(F.U.data), size(F.U, 2), F.U.l, F.U.u)

  d_A = CuArray(A)
  d_B = CuArray(B)

  d_A, d_ipiv = CUSOLVER.getrf!(d_A)
  d_B         = CUSOLVER.getrs!('N', d_A, d_ipiv, d_B)


  @test Array(d_B) ≈ A\B

  ## dA = CuSparseMatrixCSR(sA)
  ## dB = CuArray(B)
  ## dX = similar(dB)
  ## tol = 1e-6
  ## dX = CUSOLVER.csrlsvlu!(dA,dB,dX,tol,one(Cint),'O')
  ## dX = CUSOLVER.csrlsvqr!(dA,dB,dX,tol,one(Cint),'O')

end

let

  N = 5
  Nq = N + 1
  Nfields = 5
  Ne_vert = 8
  Ne_horz = 6*6*6

  b = rand(Nq, Nq, Nq, Nfields, Ne_vert, Ne_horz)

  m = n = Nq * Nfields * Ne_vert
  p = q = Nq * Nfields

  A = Array(brand(Float64, m, n, p, q))
  F = lu(A, Val(false))

  display(spy(A))
  display(spy(F.L))
  display(spy(F.U))

end
