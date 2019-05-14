using CUDAdrv
using CUDAnative

mlock!(p::Ptr, size::Integer) = ccall(:mlock, Cint, (Ptr{Cvoid}, Csize_t), p, size)
munlock!(p::Ptr, size::Integer) = ccall(:unmlock, Cint, (Ptr{Cvoid}, Csize_t), p, size)

function pin!(a)
    ad = Mem.register(Mem.Host, pointer(a), sizeof(a))
    finalizer(_ -> Mem.unregister(ad), a)
end

"""
    empiricalbandwidth(nbytes=2*1024^3; devicenumber=0, numtests=10)

Compute the emperical bandwidth in GB/s of a CUDA device using `nbytes` of
memory.  The device to test can be slected with `devicenumber` and the
bandwidth is an average of `ntests`.
"""
function empiricalbandwidth(nbytes=1024^2; devicenumber=0, ntests=10, pinned=false)
    device!(devicenumber)
    stm = CuStream(CUDAdrv.STREAM_NON_BLOCKING)

    a = rand(Char, nbytes÷sizeof(Char))
    pinned && pin!(a)

    b = Mem.alloc(Mem.Device, nbytes)

    Mem.copy!(pointer(a), b, nbytes)
    Mem.copy!(b, pointer(a), nbytes)

    t_dtoh = CUDAdrv.@elapsed stm for n = 1:ntests
        Mem.copy!(pointer(a), b, nbytes, async=true, stream=stm)
    end
    bandwidth_dtoh = nbytes*ntests/(t_dtoh*1e9)

    t_htod = CUDAdrv.@elapsed stm for n = 1:ntests
        Mem.copy!(b, pointer(a), nbytes, async=true, stream=stm)
    end
    bandwidth_htod = nbytes*ntests/(t_htod*1e9)

    Mem.free(b)

    (bandwidth_dtoh, bandwidth_htod)
end
