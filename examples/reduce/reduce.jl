using CUDAdrv, CuArrays, CUDAnative, GPUArrays
CuArrays.allowscalar(false)
import Base.Broadcast: Broadcasted, ArrayStyle
using LazyArrays

function mymap_knl!(f, dest, src)
  I = CuArrays.@cuindex dest
  @inbounds dest[I...] = f(src[I...])
  nothing
end

function _mymap!(f, dest, src)
  dev = CUDAdrv.device()
  thr = attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
  blk = length(dest) ÷ thr + 1
  @cuda blocks=blk threads=thr mymap_knl!(f, dest, src)
  return dest
end

mymap!(f, dest::CuArray, src::CuArray) = _mymap!(f, dest, src)

function mymap!(f, dest, bc::Broadcasted{ArrayStyle{CuArray}})
  axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
  bc′ = Broadcast.preprocess(dest, bc)
  _mymap!(f, dest, bc)
end

a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (1, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

mymap!(identity, d_c, @~ d_a .* d_b)

c = a .* b
c ≈ d_c

mymap!(x->x^2, d_c, d_a)
(a.^2) ≈ d_c
