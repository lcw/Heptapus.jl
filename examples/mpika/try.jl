using KernelAbstractions
using  CUDAapi
if CUDAapi.has_cuda_gpu()
    using CuArrays
    using CUDAdrv
    CuArrays.allowscalar(false)
else
    exit()
end
using MPI
using Test

device(A) = typeof(A) <: Array ? CPU() : CUDA()

function mpiyield()
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
    yield()
end

function __Irecv!(request, buf, src, tag, comm)
    request[1] = MPI.Irecv!(buf, src, tag, comm)
end

function __Isend!(request, buf, dst, tag, comm)
    request[1] = MPI.Isend(buf, dst, tag, comm)
end

function __testall!(requests; dependencies=nothing)
    done = false
    while !done
        done, _ = MPI.Testall!(requests)
        yield()
    end
end

function exchange!(d_send_buf, h_send_buf, d_recv_buf, h_recv_buf, src_rank,
                   dst_rank, send_request, recv_request, comm;
                   dependencies=nothing)

    event = Event(__Irecv!, recv_request, h_recv_buf, src_rank, 666, comm;
                  dependencies = dependencies, progress=mpiyield)
    event = Event(__testall!, recv_request; dependencies = event, progress=mpiyield)
    recv_event = async_copy!(device(d_recv_buf), d_recv_buf, h_recv_buf;
                             dependencies = event, progress=mpiyield)

    event = async_copy!(device(d_send_buf), h_send_buf, d_send_buf;
                             dependencies = dependencies, progress=mpiyield)
    event = Event(__Isend!, send_request, h_send_buf, dst_rank, 666, comm;
                  dependencies = event, progress=mpiyield)
    send_event = Event(__testall!, send_request; dependencies = event, progress=mpiyield)

    return recv_event, send_event
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    MPI.Barrier(comm)

    dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
    src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

    T = Int64
    M = 10

    d_send_buf = CuArrays.zeros(T, M)
    d_recv_buf = CuArrays.zeros(T, M)
    h_send_buf = zeros(T, M)
    h_recv_buf = zeros(T, M)

    fill!(d_send_buf, MPI.Comm_rank(comm))
    fill!(d_recv_buf, -1)
    fill!(h_send_buf, -1)
    fill!(h_recv_buf, -1)

    send_request = fill(MPI.REQUEST_NULL, 1)
    recv_request = fill(MPI.REQUEST_NULL, 1)

    synchronize()

    recv_event, send_event = exchange!(d_send_buf, h_send_buf, d_recv_buf,
                                       h_recv_buf, src_rank, dst_rank,
                                       send_request, recv_request, comm)
    wait(CUDA(), recv_event)
    wait(CUDA(), send_event)

    @test all(d_recv_buf .== src_rank)
end

main()
