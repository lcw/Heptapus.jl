SRCDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# nvcc -G -ptx

deps.jl:
	julia --project build_LLVM.v6.0.1.jl

CLANG := $(SRCDIR)/usr/tools/clang
LLC := $(SRCDIR)/usr/tools/llc
LINK := $(SRCDIR)/usr/tools/llvm-link
OPT := $(SRCDIR)/usr/tools/opt
CUDA_PATH ?= /usr/lib/cuda

volumerhs: volumerhs.cu | deps.jl
	$(CLANG) -O3 -L$(CUDA_PATH)/lib64 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64,--enable-new-dtags -o $@ $< 

volumerhs.ll: volumerhs.cu | deps.jl
	$(CLANG) -O3 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -S -emit-llvm --cuda-device-only -o $@ $< 

volumerhs.opt.ll: volumerhs.ll | deps.jl
	$(OPT) -O3 -o $@ -S $<

volumerhs.bc: volumerhs.ll | deps.jl
	$(LINK) $< $(CUDA_PATH)/nvvm/libdevice/libdevice.10.bc -o $@.link
	$(OPT) -nvvm-reflect -O3 $@.link -o $@
	rm $@.link

volumerhs.ptx: volumerhs.bc | deps.jl
	$(LLC) -O3 -mcpu=sm_70 -o $@ $<

volumerhs.o: volumerhs.ptx
	$(CUDA_PATH)/bin/ptxas -arch=sm_70 -lineinfo -v -warn-lmem-usage -warn-spills -o $@ $<

volumerhs.S: volumerhs.o
	$(CUDA_PATH)/bin/nvdisasm -sf -c -plr -g -lrm narrow $< > $@

volumerhs_smaller: volumerhs_smaller.cu | deps.jl
	$(CLANG) -O3 -L$(CUDA_PATH)/lib64 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64,--enable-new-dtags -o $@ $< 

volumerhs_smaller.ll: volumerhs_smaller.cu | deps.jl
	$(CLANG) -O3 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -S -emit-llvm --cuda-device-only -o $@ $< 

volumerhs_smaller.opt.ll: volumerhs_smaller.ll | deps.jl
	$(OPT) -O3 -o $@ -S $<

volumerhs_smaller.bc: volumerhs_smaller.ll | deps.jl
	$(LINK) $< $(CUDA_PATH)/nvvm/libdevice/libdevice.10.bc -o $@.link
	$(OPT) -nvvm-reflect -O3 $@.link -o $@
	rm $@.link

volumerhs_smaller.ptx: volumerhs_smaller.bc | deps.jl
	$(LLC) -O3 -mcpu=sm_70 -o $@ $<

volumerhs_smaller.o: volumerhs_smaller.ptx
	$(CUDA_PATH)/bin/ptxas -arch=sm_70 -lineinfo -v -warn-lmem-usage -warn-spills -o $@ $<

volumerhs_smaller.S: volumerhs_smaller.o
	$(CUDA_PATH)/bin/nvdisasm -sf -c -plr -g -lrm narrow $< > $@

clean:
	rm -f volumerhs.{o,S,bc,ptx,ll}
	rm -f volumerhs_smaller.{o,S,bc,ptx,ll}

