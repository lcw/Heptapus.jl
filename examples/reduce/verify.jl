include("reduce.jl")

len = 10^7
input = ones(Int32, len)
output = similar(input)
cpu_val = reduce(+, input)

gpu_input = CuArray(input)
gpu_output = CuArray(output)
gpu_reduce(+, gpu_input, gpu_output)
gpu_val = Array(gpu_output)[1]
@assert cpu_val == gpu_val



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
