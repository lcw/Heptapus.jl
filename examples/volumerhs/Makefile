SRCDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# nvcc -G -ptx

deps.jl:
	julia build_LLVM.v6.0.1.jl

CLANG := $(SRCDIR)/usr/tools/clang
LLC := $(SRCDIR)/usr/tools/llc
LINK := $(SRCDIR)/usr/tools/llvm-link
OPT := $(SRCDIR)/usr/tools/opt
CUDA_PATH ?= /usr/lib/cuda

vanilladeriv: vanilladeriv.cu | deps.jl
	$(CLANG) -O3 -L$(CUDA_PATH)/lib64 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -lcudart -Wl,-rpath=$(CUDA_PATH)/lib64,--enable-new-dtags -o $@ $< 

vanilladeriv.ll: vanilladeriv.cu | deps.jl
	$(CLANG) -O3 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -S -emit-llvm --cuda-device-only -o $@ $< 

vanilladeriv.bc: vanilladeriv.ll | deps.jl
	$(LINK) $< $(CUDA_PATH)/nvvm/libdevice/libdevice.10.bc -o $@.link
	$(OPT) -nvvm-reflect -O3 $@.link -o $@
	rm $@.link

vanilladeriv.ptx: vanilladeriv.bc | deps.jl
	$(LLC) -O3 -mcpu=sm_70 -o $@ $<

vanilladeriv.o: vanilladeriv.ptx
	$(CUDA_PATH)/bin/ptxas -arch=sm_70 -lineinfo -v -warn-lmem-usage -warn-spills -o $@ $<

vanilladeriv.S: vanilladeriv.o
	$(CUDA_PATH)/bin/nvdisasm -sf -c -plr -g -lrm narrow $< > $@

clean:
	rm -f vanilladeriv.{o,S,bc,ptx,ll}

