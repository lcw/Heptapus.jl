using CUDAdrv
using CUDAnative

"""
    empiricalbandwidth(nbytes=2*1024^3; devicenumber=0, numtests=10)

Compute the emperical bandwidth in GB/s of a CUDA device using `nbytes` of
memory.  The device to test can be slected with `devicenumber` and the
bandwidth is an average of `ntests`.
"""
function empiricalbandwidth(nbytes=1024^2; devicenumber=0, ntests=10, pinned=false)
    device!(devicenumber)
    stm = CuStream(CUDAdrv.STREAM_NON_BLOCKING)

    d = rand(Char, nbytes)
    a = pinned ? Mem.alloc(Mem.Host, nbytes) : pointer(d)
    b = Mem.alloc(Mem.Device, nbytes)

    Mem.copy!(a, b, nbytes)
    Mem.copy!(b, a, nbytes)

    t_dtoh = CUDAdrv.@elapsed stm for n = 1:ntests
        Mem.copy!(a, b, nbytes, async=true, stream=stm)
    end
    bandwidth_dtoh = nbytes*ntests/(t_dtoh*1e9)

    t_htod = CUDAdrv.@elapsed stm for n = 1:ntests
        Mem.copy!(b, a, nbytes, async=true, stream=stm)
    end
    bandwidth_htod = nbytes*ntests/(t_htod*1e9)

    pinned && Mem.free(a)
    Mem.free(b)

    (bandwidth_dtoh, bandwidth_htod)
end
