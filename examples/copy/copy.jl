using GPUifyLoops

kernel!(A::Array, B::Array) = kernel!(CPU(), A, B)
function kernel!(::Dev, A, B) where Dev
    @setup Dev
    @inbounds @loop for i in (1:size(A,1);
                              (blockIdx().x-1)*blockDim().x + threadIdx().x)
        A[i] = B[i]
    end
    nothing
end

a = rand(Float32, 10^9)
b = similar(a)
kernel!(b, a)
@show Base.summarysize(a)/10^9

@assert b == a

@static if Base.find_package("CuArrays") !== nothing

    using CuArrays, CUDAnative

    @eval function kernel!(A::CuArray, B::CuArray)
        threads = 1024
        blocks = ceil(Int, size(A,1)/threads)
        @cuda(threads=threads, blocks=blocks, kernel!(CUDA(), A, B))
    end

    ca = CuArray(a)
    cb = similar(ca)

    kernel!(cb, ca)

    @assert Array(cb) == Array(ca)

end
