# Using Cthulhu

The file `example.jl` documents how to use [Cthulhu][1] to descend into a
CUDAnative function.  Once in Cthulhu you can look at the optimized code to
make sure things are inlined.

It can be useful to look at the call graph for the code to make sure that loops
are getting inlined.  To do this for ptx requires a couple of steps.  First we
need the intermediate files which can be done using the `CUDAnative.@device_code`.
Once we have that we can run the LLVM `opt` tool from the shell to convert the
`kernel.unopt.ll` to `kernel.opt.ll`.  We either need to build Julia from
source to keep the llvm tools around or grab them

```
mkdir -P llvm/6.0.1
cd llvm/6.0.1
wget https://github.com/staticfloat/LLVMBuilder/releases/download/v6.0.1-5%2Bnowasm/LLVM.v6.0.1.x86_64-linux-gnu-gcc7.tar.gz
tar -xzvf LLVM.v6.0.1.x86_64-linux-gnu-gcc7.tar.gz
rm -f LLVM.v6.0.1.x86_64-linux-gnu-gcc7.tar.gz
cd ../..
```

We then need to setup our environment so that we can run similar passes to
Julia.  The variables below were taken from [perfenv.sh][4] of from
[MCAnalyzer][2]'s [contrib][3] directory.  The following sets up the standard
julia llvm passes (but is not exactly what CUDAnative [runs][5].

```
set -x JULIA_PATH (which julia)
set LLVM_PATH $HOME/opt/llvm/6.0.1/tools
set -x OPT $LLVM_PATH/opt
set -x OPTFLAGS -load=$JULIA_PATH/lib/libjulia.so
set -x JLPASSES -tbaa -PropagateJuliaAddrspaces -simplifycfg -dce -sroa -memcpyopt -always-inline -AllocOpt \
                -instcombine -simplifycfg -sroa -instcombine -jump-threading -instcombine -reassociate \
                -early-cse -AllocOpt -loop-idiom -loop-rotate -LowerSIMDLoop -licm -loop-unswitch \
                -instcombine -indvars -loop-deletion -loop-unroll -AllocOpt -sroa -instcombine -gvn \
                -memcpyopt -sccp -sink -instsimplify -instcombine -jump-threading -dse -AllocOpt \
                -simplifycfg -loop-idiom -loop-deletion -jump-threading -slp-vectorizer -adce \
                -instcombine -loop-vectorize -instcombine -barrier -LowerExcHandlers \
                -GCInvariantVerifier -LateLowerGCFrame -dce -LowerPTLS -simplifycfg -CombineMulAdd
```

We can look at the control flow graph with

```
julia --project=. example.jl
$OPT -dot-cfg 'tmp/kernel!_2.opt.ll'
dot -Tpdf cfg.ptxcall_kernel__3.dot -o out.pdf
```

Once this is sourced we can then optimize the LLVM ir with

```
$OPT $OPTFLAGS $JLPASSES -S -o opt.ll 'tmp/kernel!_2.opt.ll'
```

where we can turn on and off various passes.


[1]: https://github.com/JuliaDebug/Cthulhu.jl
[2]: https://github.com/vchuravy/MCAnalyzer.jl
[3]: https://github.com/vchuravy/MCAnalyzer.jl/tree/d900fd11e3c9bcd1f574e74aeaa66709273ff1b7/contrib
[4]: https://github.com/vchuravy/MCAnalyzer.jl/blob/d900fd11e3c9bcd1f574e74aeaa66709273ff1b7/contrib/perfenv.sh
[5]: https://github.com/JuliaGPU/CUDAnative.jl/blob/e28c5f07cd2dae78a41d0d67ea2dfd4b374a92fe/src/compiler/optim.jl#L3
