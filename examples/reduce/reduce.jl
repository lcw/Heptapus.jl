using CUDAdrv, CuArrays, CUDAnative
import Base.Broadcast: Broadcasted, ArrayStyle

# Fast parallel reduction for Kepler hardware
# - uses shuffle and shared memory to reduce efficiently
# - support for large arrays
#
# Based on devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

# Reduce a value across a warp
@inline function reduce_warp(op::F, val::T, threadmask::UInt32)::T where {F<:Function,T}
    offset = CUDAnative.warpsize() ÷ 2
    # TODO: this can be unrolled if warpsize is known...
    while offset > 0
        # TODO: uncomment when working
        # val = op(val, shfl_down_sync(val, offset, 32, threadmask))
        val = op(val, shfl_down_sync(val, offset, 32))
        offset ÷= 2
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::F, val::T, useval::Bool)::T where {F<:Function,T}
    # shared mem for partial sums
    #
    # TODO the size should be based on number of threads per block divided by
    # the warpsize.
    shared = @cuStaticSharedMem(T, 32)

    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())

    # each warp performs partial reduction
    threadmask = vote_ballot(useval)
    # if useval
    if true
        val = reduce_warp(op, val, threadmask)
    end

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # final reduce within first warp
    if wid == 1
        warpexisted = threadIdx().x <= fld1(blockDim().x, CUDAnative.warpsize())
        @inbounds val = warpexisted ? shared[lane] : zero(T)
        threadmask = vote_ballot(warpexisted)
        # if warpexisted
        if true
            val = reduce_warp(op, val, threadmask)
        end
    end

    return val
end

# Reduce an array across a complete grid
function reduce_grid(op::F,
                     input::Union{Broadcasted{ArrayStyle{CuArray}},
                                  CuDeviceArray{T}},
                     output::CuDeviceArray{T},
                     len::Integer) where {F<:Function,T}
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    start = (blockIdx().x-1) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    for i = start:step:len
        I = Tuple(CartesianIndices(axes(input))[i])
        @inbounds val = op(val, input[I...])
    end

    val = reduce_block(op, val, start <= len)

    if threadIdx().x == 1
        @inbounds output[blockIdx().x] = val
    end

    return
end

"""
Reduce a large array.

Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce(op::Function, input, output::CuArray{T}) where {T}
    if capability(device()) < v"3.0"
        @warn("this example requires a newer GPU")
        exit(0)
    end

    len = length(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((len + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        throw(ArgumentError("output array too small, should be at least $blocks elements"))
    end

    @cuda blocks=blocks threads=threads reduce_grid(op, input, output, len)
    @cuda threads=1024 reduce_grid(op, output, output, blocks)
end

# FURTHER IMPROVEMENTS:
# - use atomic memory operations
# - dynamic block/grid size based on device capabilities
# - vectorized memory access
#   devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
