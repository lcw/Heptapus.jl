using KernelAbstractions
using CuArrays
using CUDAnative
using CUDAdrv
using MPI
using KernelAbstractions: CudaEvent, Event

@kernel function kernel!(c, @Const(a), @Const(b))
  i = @index(Global)

  @inbounds a_val = a[i]
  @inbounds b_val = b[i]
  @inbounds c_val = c[i]

  for j in 1:99
    a_val = a_val * b_val + c_val
    b_val = a_val * b_val + c_val
    c_val = a_val * b_val + c_val
  end

  @inbounds c[i] = a_val * b_val + c_val

  return
end

@kernel function mpikernel!(c, b, src_rank, dst_rank)
  rreq = MPI.Irecv!(c, src_rank, 222, comm)
  sreq = MPI.Isend(b, dst_rank, 222, comm)
  stats = MPI.Waitall!([sreq, rreq])
end

function pin!(a)
  ad = Mem.register(Mem.Host, pointer(a), sizeof(a))
  finalizer(_ -> Mem.unregister(ad), a)
end

function recordevent(stream)
  event = CuEvent(CUDAdrv.EVENT_DISABLE_TIMING)
  CUDAdrv.record(event, stream)
  return CudaEvent(event)
end

function async_copyto!(destptr, srcptr, N::Integer; stream=CuDefaultStream(),
                       dependencies=nothing)
  if dependencies isa Event
    dependencies = (dependencies,)
  end
  if dependencies !== nothing
    for event in dependencies
      @assert event isa CudaEvent
      CUDAdrv.wait(event.event, stream)
    end
  end

  unsafe_copyto!(destptr, srcptr, N, async=true, stream=stream)

  return recordevent(stream)
end


function main()
  if !MPI.Initialized()
    MPI.Init()
  end
  comm = MPI.COMM_WORLD
  local_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED,
                                   MPI.Comm_rank(comm))
  device!(MPI.Comm_rank(local_comm) % length(devices()))
  copystream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
  CuArrays.allowscalar(false)

  synchronize(copystream)
  MPI.Barrier(comm)

  dst_rank = mod(MPI.Comm_rank(comm)+1, MPI.Comm_size(comm))
  src_rank = mod(MPI.Comm_rank(comm)-1, MPI.Comm_size(comm))

  T = Float32
  M = 1024
  N = M*100

  A = CuArray(rand(T, M, N))
  B = CuArray(rand(T, M, N))
  C = CuArray(rand(T, M, N))
  d = CuArray(rand(T, M))

  b = Array{T}(undef, M)
  c = Array{T}(undef, M)
  pin!(b)
  pin!(c)

  len = length(A)
  threads = min(len, 1024)
  blocks = len ÷ threads

  compevent = recordevent(copystream)
  copyevent = recordevent(copystream)
  for j = 1:100
    copyevent = async_copyto!(pointer(b), pointer(B), M, stream=copystream, dependencies=compevent)
    compevent = kernel!(CUDA(), 256)(C, A, B, ndrange=length(C), dependencies=compevent)
    copyevent = mpikernel!(CPU(), 1)(c, b, src_rank, dst_rank, ndrange=1, dependencies=copyevent)
    copyevent = async_copyto!(pointer(d), pointer(c), M, stream=copystream, dependencies=copyevent)
    copyevent = async_copyto!(pointer(C), pointer(d), M, stream=copystream, dependencies=copyevent)
    compevent = kernel!(CUDA(), 256)(C, A, B, ndrange=length(C), dependencies=copyevent)
  end
end
main()
