using LinearAlgebra, GPUifyLoops, StaticArrays, StructArrays, Random

Random.seed!(4)

"""
Implementation of the 2 x 2 eignvalue and eigenvector solver found in:
  @misc{Borges2017,
    title = {Numerically sane solution of the 2 x 2 real symmetric eigenvalue problem},
    author = {Carlos F. Borges},
    year = {2017},
    url = {https://calhoun.nps.edu/handle/10945/51506},
  }
"""
@inline function cfbeigen(A::SHermitianCompact{2,T,3}) where {T <: Real}
    α₁ = A[1, 1]
    α₂ = A[2, 2]
    β = A[2, 1]

    δ = (α₁ - α₂)/2
    h = hypot(δ, β)

    if h == 0
        c = one(T)
        s = zero(T)
    elseif δ ≥ 0
        c = δ + h
        s = β
    else
        c = β
        s = h - δ
    end

    ρ = 1/hypot(s, c)

    c = ρ*c
    s = ρ*s

    V = @SMatrix [c -s;
                  s  c]
    λ = @SVector [c^2*α₁ + s*(s*α₂ + 2c*β),
                  c^2*α₂ + s*(s*α₁ - 2c*β)]

    return Eigen(λ, V)
end

# initialize with random symmetric matrices
s = StructArray(SHermitianCompact(@SVector(rand(3))) for i=1:5)

F1 = eigen(s[1])
F2 = cfbeigen(s[1])

@info "Test" F1 F2
