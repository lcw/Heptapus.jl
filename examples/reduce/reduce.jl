using CuArrays, CUDAnative, GPUArrays
CuArrays.allowscalar(false)
import Base.Broadcast: Broadcasted, ArrayStyle
using LazyArrays

a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (1, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)
d_c .= NaN

function mycopyto!(dest, bc::Broadcasted{Nothing})
  axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
  bc′ = Broadcast.preprocess(dest, bc)
  gpu_call(dest, (dest, bc′)) do state, dest, bc′
    let I = CartesianIndex(@cartesianidx(dest))
      @inbounds dest[I] = bc′[I]
    end
    return
  end
  return dest
end

# Base defines this method as a performance optimization, but we don't know how to do
# `fill!` in general for all `GPUDestArray` so we just go straight to the fallback
@inline mycopyto!(dest, bc::Broadcasted{ArrayStyle{CuArray}}) =
    mycopyto!(dest, convert(Broadcasted{Nothing}, bc))

mycopyto!(d_c, @~ d_a .* d_b)

c = a .* b
c ≈ d_c
