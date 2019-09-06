using LinearAlgebra, GPUifyLoops, StaticArrays, StructArrays, Random

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays, CUDAnative
end


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
    @inbounds α₁ = A[1, 1]
    @inbounds α₂ = A[2, 2]
    @inbounds β = A[2, 1]

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
N = 5
T = Float64

SH = SHermitianCompact{2, T, 3}
As = StructArray{SH}(undef, N, unwrap = t -> t <: Union{SVector,Tuple})
for i = 1:N
    As[i] = SHermitianCompact(@SVector(rand(3)))
end

SM = SMatrix{2, 2, T, 4}
Vs = StructArray{SM}(undef, N, unwrap = t -> t <: Union{NTuple})

SV = SVector{2, T}
λs = StructArray{SV}(undef, N, unwrap = t -> t <: Union{NTuple})

function kernel(As, Vs, λs)
    @inbounds @loop for i in (1:size(As,1);
                              (blockIdx().x-1)*blockDim().x + threadIdx().x)
        F = cfbeigen(As[i])
        Vs[i] = F.vectors
        λs[i] = F.values
    end
    nothing
end

function arrayskernel(As, Vs, λs)
    @inbounds @loop for i in (1:size(As[1],1);
                              (blockIdx().x-1)*blockDim().x + threadIdx().x)
        A = SHermitianCompact(@SVector([As[1][i], As[2][i], As[3][i]]))

        F = cfbeigen(A)

        @unroll for j = 1:length(F.vectors)
            Vs[j][i] = F.vectors[j]
        end

        @unroll for j = 1:length(F.values)
            λs[j][i] = F.values[j]
        end
    end
    nothing
end

@launch CPU() kernel(As, Vs, λs)

display(As)
display(Vs)
display(λs)

@static if Base.find_package("CuArrays") !== nothing
    cAs = CuArray.(fieldarrays(As.lowertriangle.data))
    cVs = similar.(CuArray.(fieldarrays(Vs.data)))
    cλs = similar.(CuArray.(fieldarrays(λs.data)))

    threads = 1024
    blocks = cld(size(cAs[1],1), threads)
    @launch(CUDA(), threads=threads, blocks=blocks, arrayskernel(cAs, cVs, cλs))

    display(cAs)
    display(cVs)
    display(cλs)
end
