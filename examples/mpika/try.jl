# EXCLUDE FROM TESTING
using KernelAbstractions
using  CUDAapi
if CUDAapi.has_cuda_gpu()
    using CUDAdrv
    using CuArrays
    CuArrays.allowscalar(false)
else
    exit()
end
using MPI
using Test

device(A) = typeof(A) <: Array ? CPU() : CUDA()

@kernel function kernel!(c, a, b)
  I = @index(Global)
  @inbounds a_val = a[I]
  @inbounds b_val = b[I]
  @inbounds c_val = c[I]

  for j in 1:33
    a_val = a_val * b_val + c_val
    b_val = a_val * b_val + c_val
    c_val = a_val * b_val + c_val
  end

  @inbounds c[I] = a_val * b_val + c_val
end

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
                  dependencies = dependencies, progress=mpiyield)
    send_event = Event(__testall!, send_request; dependencies = event, progress=mpiyield)

    return recv_event, send_event
end

function init(T, M, comm)
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

    return (d_send_buf = d_send_buf,
            d_recv_buf = d_recv_buf,
            h_send_buf = h_send_buf,
            h_recv_buf = h_recv_buf,
            send_request = send_request,
            recv_request = recv_request,
            comm = MPI.Comm_dup(comm))
end

function main()
    if !MPI.Initialized()
        MPI.Init()
    end
    comm = MPI.COMM_WORLD
    MPI.Barrier(comm)

    dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
    src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

    T = Float64
    M = 10
    N = 1024^2

    a = CuArrays.rand(T, N)
    b = CuArrays.rand(T, N)
    c = CuArrays.rand(T, N)

    e1 = init(T, M, comm)
    e2 = init(T, M, comm)

    pc_kernel! = kernel!(CUDA(), 256)
    KernelAbstractions.precompile(pc_kernel!, c, a, b; ndrange=(length(c),))

    synchronize()

    event = Event(CUDA())

    @info "Here 1"
    recv_e1, send_e1 = exchange!(e1.d_send_buf, e1.h_send_buf,
                                 e1.d_recv_buf, e1.h_recv_buf,
                                 src_rank, dst_rank,
                                 e1.send_request, e1.recv_request,
                                 e1.comm; dependencies=event)

    @info "Here 3"
    event = kernel!(CUDA(), 256)(c, a, b, ndrange=length(c), dependencies = event)

    @info "Here 4"
    wait(recv_e1)

    @info "Here 5"
    @test all(e1.d_recv_buf .== src_rank)

    @info "Here 6"
    wait(send_e1)

    @info "Here 7"
    wait(event)
end

main()
