module Heptapus

using TypedTables, CSV, CUDAdrv

export empiricalbandwidth, Roofline

function transfer!(A, B, N, stream)
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        unsafe_copyto!(destptr, srcptr, N, async=true, stream=stream)
    end
end

"""
    empiricalbandwidth(nbytes=2*1024^3; devicenumber=0, numtests=10)

Compute the emperical bandwidth in GB/s of a CUDA device using `nbytes` of
memory.  The device to test can be slected with `devicenumber` and the
bandwidth is an average of `ntests`.
"""
function empiricalbandwidth(nbytes=2*1024^3; devicenumber=0, ntests=10)
    dev = CuDevice(devicenumber)
    ctx = CuContext(dev)
    stream = CuStream(; flags = CUDAdrv.STREAM_NON_BLOCKING) 

    a = Mem.alloc(Mem.DeviceBuffer, nbytes÷2)
    b = Mem.alloc(Mem.DeviceBuffer, nbytes÷2)

    transfer!(a, b, nbytes÷2, stream)
    transfer!(b, a, nbytes÷2, stream)

    t = CUDAdrv.@elapsed stream for n = 1:ntests
        transfer!(a, b, nbytes÷2, stream)
        transfer!(b, a, nbytes÷2, stream)
    end

    Mem.free(a)
    Mem.free(b)

    bandwidth = 2*nbytes*ntests/(t*1e9)
end


"""
    Roofline(command::Cmd)

Use `nvprof` to profile `command` and compute for each kernel executed:

  - arithmetic intensity;
  - performance (GFLOP/s);
  - kernel max empirical bandwidth (GB/s);
  - max GFLOP/s estimate (GFLOP/s);
  - max empirical bandwidth (GB/s);
  - local loads or stores (`Bool`);
  - floaing point type used; and
  - mixed floating point operations (`Bool`).
"""
struct Roofline
    t::Table

    function Roofline(command::Cmd)
        s = mktemp() do f, _
            metrics = [:dram_read_throughput,
                       :dram_write_throughput,
                       :dram_write_transactions,
                       :dram_read_transactions,
                       :local_load_transactions,
                       :local_store_transactions,
                       :flop_hp_efficiency,
                       :flop_sp_efficiency,
                       :flop_dp_efficiency,
                       :flop_count_hp,
                       :flop_count_sp,
                       :flop_count_dp]
            cmd = `nvprof -u ms --csv --metrics $(join(metrics,",")) --log-file $f $command`
            @info "Getting metrics" cmd
            run(cmd)
            Table(CSV.File(f, comment="=", allowmissing=:none))
        end

        t = mktemp() do f, _
            cmd = `nvprof -u ms --csv --log-file $f $command`
            @info "Getting timings" cmd
            run(cmd)
            Table(CSV.File(f, comment="=", allowmissing=:none, datarow=3))
        end

        kernels = unique(s.Kernel)

        @info "Kernels found in timings" kernels
        @info "Kernels found in measurements" unique(t.Name)

        if !(kernels ⊆ unique(t.Name))
          error("Kernel names do not match")
        end

        getmetric(T, k, m; trim=0) =
            parse(T, s[map(row -> row.Kernel == k &&
                           getproperty(row, Symbol("Metric Name")) == String(m),
                           s)][1].Avg[1:end-trim])

        # get average kernel execution time in seconds
        gettime(k) = t[map(row -> row.Name == k, t)][1].Avg/1e3

        eb = empiricalbandwidth()
        @info "Maximum empirical bandwidth $eb (GB/s)"

        maxempiricalbandwidth = fill(eb, length(kernels))
        arithmeticintensity = zeros(length(kernels))
        performance = similar(arithmeticintensity)
        kernelmaxempiricalbandwidth = similar(arithmeticintensity)
        maxgflopsestimate = similar(arithmeticintensity)
        haslocal = zeros(Bool, length(kernels))
        hasmixedflops = zeros(Bool, length(kernels))
        floptype = Array{Type}(undef, length(kernels))

        @info "Computing results"
        for (i, k) in enumerate(kernels)
            dram_write_throughput = getmetric(Float64, k, :dram_write_throughput, trim=4)
            dram_read_throughput = getmetric(Float64, k, :dram_read_throughput, trim=4)

            dram_write_transactions = getmetric(Int64, k, :dram_write_transactions)
            dram_read_transactions = getmetric(Int64, k, :dram_read_transactions)

            local_load_transactions = getmetric(Int64, k, :local_load_transactions)
            local_store_transactions = getmetric(Int64, k, :local_store_transactions)

            flop_efficiency = (hp=getmetric(Float64, k, :flop_hp_efficiency, trim=1),
                               sp=getmetric(Float64, k, :flop_sp_efficiency, trim=1),
                               dp=getmetric(Float64, k, :flop_dp_efficiency, trim=1))

            flop_count = (hp=getmetric(Int64, k, :flop_count_hp),
                          sp=getmetric(Int64, k, :flop_count_sp),
                          dp=getmetric(Int64, k, :flop_count_dp))

            elapsedtime = gettime(k)

            if local_load_transactions > 0 || local_store_transactions > 0
                haslocal[i] = true
                @warn """Kernel $k has nonzero load or store transactions.

                    This could be evidece of register spilling and the returned
                    roofline numbers will not be accurate for this kernel.

                """ local_load_transactions local_store_transactions
            end

            if sum(flop_count) != maximum(flop_count)
                hasmixedflops[i] = true
                @warn """Kernel $k has floating point operations for multiple types.

                    The performance results will use the floating point type with
                    the maximum flop count.  The may or may not be what is desired.

                    Note: `flop_count` contains half, single, and double precision.
                """ flop_count
            end

            p = argmax(flop_count)
            floptype[i] = (hp=Float16, sp=Float32, dp=Float64)[p]

            # Note that each transaction is 32 bytes
            bytes = 32(dram_write_transactions + dram_read_transactions)
            flops = flop_count[p]

            arithmeticintensity[i] = flops/bytes

            # in GFLOP/s
            performance[i] = (flops/elapsedtime)/1e9
            maxgflopsestimate[i] = 100performance[i]/flop_efficiency[p]

            kernelmaxempiricalbandwidth[i] = empiricalbandwidth(bytes)
        end

        new(Table(kernels = kernels,
                  arithmeticintensity = arithmeticintensity,
                  performance = performance,
                  kernelmaxempiricalbandwidth = kernelmaxempiricalbandwidth,
                  maxgflopsestimate = maxgflopsestimate,
                  maxempiricalbandwidth = maxempiricalbandwidth,
                  haslocal = haslocal,
                  floptype = floptype,
                  hasmixedflops = hasmixedflops))
    end
end

function Base.show(io::IO, r::Roofline)
    print(io, "Roofline containing ")
    show(io, MIME"text/plain"(), r.t)
end

end # module
