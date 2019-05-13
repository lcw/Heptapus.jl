# fish script that setups an environment suitable to investigate various codegen issues
#
# borrowed from
#   https://github.com/vchuravy/MCAnalyzer.jl/tree/master/contrib

###
# Random notes:
# Use `$OPT $OPTFLAGS $JLPASSES -S -o opt.ll src.ll` to run the julia passes 
# Use `$OPT -dot-cfg opt.ll` to get a control flow graph, to render use:
# `dot -Tpdf my.dot -o my.pdf`
#
# TODO: Probably better to integrate with `-print-before`

    set JULIA_PATH $HOME/julia/usr
set LLVM_PATH $HOME/Heptapus.jl/examples/volumerhs-small/usr/tools

if test -e $JULIA_PATH/bin/julia-debug
    set SUFFIX "-debug"
else
    set SUFFIX ""
end

set -x JL $JULIA_PATH/bin/julia$SUFFIX
set -x OPT $LLVM_PATH/opt
set -x LLC $LLVM_PATH/llc
set -x OPTFLAGS -load=$JULIA_PATH/lib/libjulia$SUFFIX.so

# Approximated Clang pipeline
# set -x JLPASSES -domtree -mem2reg -deadargelim -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq \
#                 -opt-remark-emitter -instcombine -simplifycfg -basiccg -globals-aa -prune-eh -inline -functionattrs \
#                 -argpromotion -domtree -sroa -basicaa -aa -memoryssa -early-cse-memssa -domtree -basicaa -aa -lazy-value-info -jump-threading \
#                 -lazy-value-info -correlated-propagation -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq \
#                 -opt-remark-emitter -instcombine -libcalls-shrinkwrap -loops -branch-prob -block-freq -lazy-branch-prob -lazy-block-freq \
#                 -opt-remark-emitter -pgo-memop-opt -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter \
#                 -tailcallelim -simplifycfg -reassociate -domtree -loops -loop-simplify -lcssa-verification -lcssa -basicaa -aa -scalar-evolution \
#                 -loop-rotate -licm -loop-unswitch -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter \
#                 -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -indvars -loop-idiom -loop-deletion -loop-unroll -mldst-motion \
#                 -aa -memdep -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -gvn -basicaa -aa -memdep -memcpyopt -sccp -domtree -demanded-bits -bdce \
#                 -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -lazy-value-info -jump-threading -lazy-value-info \
#                 -correlated-propagation -domtree -basicaa -aa -memdep -dse -loops -loop-simplify -lcssa-verification -lcssa -aa -scalar-evolution -licm \
#                 -postdomtree -adce -simplifycfg -domtree -basicaa -aa -loops -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -barrier \
#                 -elim-avail-extern -basiccg -rpo-functionattrs -globalopt -globaldce -basiccg -globals-aa -float2int -domtree -loops -loop-simplify -lcssa-verification \
#                 -lcssa -basicaa -aa -scalar-evolution -loop-rotate -loop-accesses -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -loop-distribute \
#                 -branch-prob -block-freq -scalar-evolution -basicaa -aa -loop-accesses -demanded-bits -lazy-branch-prob -lazy-block-freq -opt-remark-emitter \
#                 -loop-vectorize -loop-simplify -scalar-evolution -aa -loop-accesses -loop-load-elim -basicaa -aa -lazy-branch-prob -lazy-block-freq \
#                 -opt-remark-emitter -instcombine -simplifycfg -domtree -loops -scalar-evolution -basicaa -aa -demanded-bits -lazy-branch-prob -lazy-block-freq \
#                 -opt-remark-emitter -slp-vectorizer -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -loop-unroll \
#                 -lazy-branch-prob -lazy-block-freq -opt-remark-emitter -instcombine -loop-simplify -lcssa-verification -lcssa -scalar-evolution -licm \
#                 -alignment-from-assumptions -strip-dead-prototypes -globaldce -constmerge -domtree -loops -branch-prob -block-freq -loop-simplify \
#                 -lcssa-verification -lcssa -basicaa -aa -scalar-evolution -branch-prob -block-freq -loop-sink -lazy-branch-prob -lazy-block-freq \
#                 -opt-remark-emitter -instsimplify -div-rem-pairs -simplifycfg 

set -x JLPASSES -tbaa -PropagateJuliaAddrspaces -simplifycfg -dce -sroa -memcpyopt -always-inline -AllocOpt \
          -instcombine -simplifycfg -sroa -instcombine -jump-threading -instcombine -reassociate \
          -early-cse -AllocOpt -loop-idiom -LowerSIMDLoop -loop-unroll -barrier -loop-rotate -licm -loop-unswitch \
          -instcombine -indvars -loop-deletion -loop-unroll -AllocOpt -sroa -instcombine -gvn \
          -memcpyopt -sccp -sink -instsimplify -instcombine -jump-threading -dse -AllocOpt \
          -simplifycfg -loop-idiom -loop-deletion -jump-threading -slp-vectorizer -adce \
          -instcombine -loop-vectorize -instcombine -barrier -LowerExcHandlers \
          -GCInvariantVerifier -LateLowerGCFrame -dce -LowerPTLS -simplifycfg -CombineMulAdd

set -x JLPASSES_TRUNC -tbaa -PropagateJuliaAddrspaces -simplifycfg -dce -sroa -memcpyopt -always-inline -AllocOpt \
          -LowerSIMDLoop -barrier -LowerExcHandlers \
          -GCInvariantVerifier -LateLowerGCFrame -dce -LowerPTLS -simplifycfg

# This one is a truncated pass pipeline I often use to debug the loop vectorizer
set -x JLPASSES_UNTIL_LV -tbaa -PropagateJuliaAddrspaces -simplifycfg -dce -sroa -memcpyopt -always-inline -AllocOpt \
          -instcombine -simplifycfg -sroa -instcombine -jump-threading -instcombine -reassociate \
          -early-cse -AllocOpt -loop-idiom -loop-rotate -LowerSIMDLoop -licm -loop-unswitch \
          -instcombine -indvars -loop-deletion -loop-unroll -AllocOpt -sroa -instcombine -gvn \
          -memcpyopt -sccp -sink -instsimplify -instcombine -jump-threading -dse -AllocOpt \
          -simplifycfg -loop-idiom -loop-deletion -jump-threading -slp-vectorizer -adce \
          -instcombine
