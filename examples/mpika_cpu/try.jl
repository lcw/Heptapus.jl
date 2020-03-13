using MPI
using KernelAbstractions
using KernelAbstractions: CPUEvent, Event

function mpiyield()
  MPI.Iprobe(MPI.MPI_ANY_SOURCE, MPI.MPI_ANY_TAG, MPI.COMM_WORLD)
  yield()
end

function __testall!(requests; dependencies=nothing)
  wait(CPU(), MultiEvent(dependencies), mpiyield)

  done = false
  while !done
    @show done, _ = MPI.Testall!(requests)
    yield()
  end
end

function exchange!(a, b, c, d, src_rank, dst_rank, sendreq, recvreq, comm; dependencies=nothing)
  event = CPUEvent(@async(__nonexistent!()))
  event1 = CPUEvent(@async(__testall!(recvreq; dependencies=event1)))
  event2 = CPUEvent(@async(__testall!(sendreq; dependencies=dependencies)))

  return MultiEvent((event1, event2))
end

function main()
  if !MPI.Initialized()
    MPI.Init()
  end
  comm = MPI.COMM_WORLD
  MPI.Barrier(comm)

  dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
  src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

  T = Float32
  M = 1024

  a = fill(T, MPI.Comm_rank(comm))
  b = Array{T}(undef, M)
  c = Array{T}(undef, M)
  d = Array{T}(undef, M)

  sendreq = fill(MPI.REQUEST_NULL, 1)
  recvreq = fill(MPI.REQUEST_NULL, 1)

  event = exchange!(a, b, c, d, src_rank, dst_rank, sendreq, recvreq, comm; dependencies=nothing)
  wait(event)
end
main()
