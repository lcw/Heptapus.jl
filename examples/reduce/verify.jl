using CuArrays, LazyArrays
CuArrays.allowscalar(false)

include("reduce.jl")

len = 10^7 + 13
input = ones(Int32, len)
output = similar(input, 1024)
cpu_val = reduce(+, input)

gpu_input = CuArray(input)
gpu_output = CuArray(output)

gpu_reduce(+, gpu_input, gpu_output)
gpu_val = Array(gpu_output)[1]
@assert cpu_val == gpu_val

gpu_reduce(+, (@~ gpu_input .+ gpu_input), gpu_output)
gpu_val = Array(gpu_output)[1]
cpu_val = reduce(+, (@~ input .+ input))
@assert cpu_val == gpu_val

a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (1, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a, 1024)

gpu_reduce(+, (@~ d_a .* d_b), d_c)
gpu_val = Array(d_c)[1]
@assert gpu_val == reduce(+, (@~ a .* b))
