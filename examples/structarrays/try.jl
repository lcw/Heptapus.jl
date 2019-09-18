using CUDAnative, CuArrays, StaticArrays, StructArrays

CUDAnative.cudaconvert(s::StructArray) = replace_storage(cudaconvert, s)

# c = [(a=SHermitianCompact(@SVector(rand(3))), b=(@SVector(rand(2)))) for i=1:5]

c = [SHermitianCompact(@SVector(rand(3))) for i=1:5]
d = StructArray(c, unwrap = t -> t <: Union{SHermitianCompact,SVector,Tuple})

dd = replace_storage(CuArray, d)
de = similar(dd)

@show typeof(dd)

function kernel!(dest, src)
  i = (blockIdx().x-1)*blockDim().x + threadIdx().x
  if i <= length(dest)
    dest[i] = src[i]
  end
  return nothing
end

threads = 1024
blocks = cld(length(dd),threads)

@cuda threads=threads blocks=blocks kernel!(de, dd)
