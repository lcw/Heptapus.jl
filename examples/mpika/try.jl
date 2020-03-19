using KernelAbstractions
using MPI

function mpiyield()
    MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
    yield()
end

function __Irecv!(request, buf, src, tag, comm)
    @info "Recv" buf src tag
    request[1] = MPI.Irecv!(buf, src, tag, comm)
end

function __Isend!(request, buf, dst, tag, comm)
    @info "Send" buf dst tag
    request[1] = MPI.Isend!(buf, dst, tag, comm)
end

function __testall!(requests; dependencies=nothing)
    @info "Testall" requests
    done = false
    while !done
        done, _ = MPI.Testall!(requests)
        yield()
    end
end

function exchange!(h_send_buf, h_recv_buf, src_rank,
                   dst_rank, send_request, recv_request, comm;
                   dependencies=nothing)

    event = Event(__Irecv!, recv_request, h_recv_buf, src_rank, 666, comm;
                  dependencies = dependencies)
    recv_event = Event(__testall!, recv_request; dependencies = event)

    event = Event(__Isend!, send_request, h_send_buf, dst_rank, 666, comm;
                  dependencies = dependencies)
    send_event = Event(__testall!, send_request; dependencies = event)

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

    h_send_buf = zeros(T, M)
    h_recv_buf = zeros(T, M)

    fill!(h_send_buf, MPI.Comm_rank(comm))
    fill!(h_recv_buf, -1)

    send_request = fill(MPI.REQUEST_NULL, 1)
    recv_request = fill(MPI.REQUEST_NULL, 1)

    event = Event(CPU())
    recv_event, send_event = exchange!(h_send_buf, h_recv_buf,
                                       src_rank, dst_rank, send_request,
                                       recv_request, comm;
                                       dependencies = event)
    wait(CPU(), recv_event, yield)
    wait(CPU(), send_event, yield)

    @assert all(h_recv_buf .== src_rank)
end

main()
