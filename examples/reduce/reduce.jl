using CUDAdrv, CuArrays, CUDAnative, GPUArrays
CuArrays.allowscalar(false)
import Base.Broadcast: Broadcasted, ArrayStyle
using LazyArrays

function mycopyto_knl!(dest, src)
  I = CuArrays.@cuindex dest
  @inbounds dest[I...] = src[I...]
  nothing
end

function _mycopyto!(dest, src)
  dev = CUDAdrv.device()
  thr = attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
  blk = length(dest) ÷ thr + 1
  @cuda blocks=blk threads=thr mycopyto_knl!(dest, src)
  return dest
end

mycopyto!(dest::CuArray, src::CuArray) = _mycopyto!(dest, src)

function mycopyto!(dest, bc::Broadcasted{ArrayStyle{CuArray}})
  axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
  bc′ = Broadcast.preprocess(dest, bc)
  _mycopyto!(dest, bc)
end

a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (1, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

mycopyto!(d_c, @~ d_a .* d_b)

c = a .* b
c ≈ d_c

mycopyto!(d_c, d_a)
a ≈ d_c
