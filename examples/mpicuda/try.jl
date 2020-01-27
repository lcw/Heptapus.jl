using CuArrays
using CUDAnative
using CUDAdrv
using MPI

function pin!(a)
  ad = Mem.register(Mem.Host, pointer(a), sizeof(a))
  finalizer(_ -> Mem.unregister(ad), a)
end

function kernel!(c, a, b)
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  @inbounds a_val = a[i]
  @inbounds b_val = b[i]
  @inbounds c_val = c[i]

  for j in 1:99
    a_val = CUDAnative.fma(a_val, b_val, c_val)
    b_val = CUDAnative.fma(a_val, b_val, c_val)
    c_val = CUDAnative.fma(a_val, b_val, c_val)
  end

  @inbounds c[i] = CUDAnative.fma(a_val, b_val, c_val)

  return
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
  compstream = CuStream(CUDAdrv.STREAM_NON_BLOCKING)
  CuArrays.allowscalar(false)

  synchronize(copystream)
  synchronize(compstream)
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

  for j = 1:100
    unsafe_copyto!(pointer(b), pointer(B), M, async=true, stream=copystream)
    @cuda threads=threads blocks=blocks stream=compstream kernel!(C, A, B)

    synchronize(copystream)

    rreq = MPI.Irecv!(c, src_rank, 222, comm)
    sreq = MPI.Isend(b, dst_rank, 222, comm)
    stats = MPI.Waitall!([sreq, rreq])

    unsafe_copyto!(pointer(d), pointer(c), M, async=true, stream=copystream)

    synchronize(copystream)
    synchronize(compstream)

    unsafe_copyto!(pointer(C), pointer(d), M, async=true, stream=copystream)
    @cuda threads=threads blocks=blocks stream=copystream kernel!(C, A, B)

    synchronize(copystream)
  end
end
main()
