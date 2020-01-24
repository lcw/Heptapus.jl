using CuArrays
using CUDAnative
using CUDAdrv
using MPI

function pin!(a)
  ad = Mem.register(Mem.Host, pointer(a), sizeof(a))
  finalizer(_ -> Mem.unregister(ad), a)
end

function main()
  if !MPI.Initialized()
    MPI.Init()
  end
  comm = MPI.COMM_WORLD
  local_comm = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED,
                                   MPI.Comm_rank(comm))
  device!(MPI.Comm_rank(local_comm) % length(devices()))
  stm = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
  CuArrays.allowscalar(false)

  MPI.Barrier(comm)

  T = Float32
  nbytes=1024^2

  a = rand(T, nbytes÷sizeof(T))
  pin!(a)

  b = CuArray{T}(undef, size(a))

  unsafe_copyto!(pointer(a), pointer(b), length(a), async=true, stream=stm)
  unsafe_copyto!(pointer(b), pointer(a), length(a), async=true, stream=stm)

  synchronize(stm)
end

main()
