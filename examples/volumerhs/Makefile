SRCDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

deps.jl:
	julia build_LLVM.v6.0.1.jl

CLANG := $(SRCDIR)/usr/tools/clang
LLC := $(SRCDIR)/usr/tools/clang
CUDA_PATH ?= /usr/lib/cuda

vanilladeriv: vanilladeriv.cu | deps.jl
	$(CLANG) -L$(CUDA_PATH)/lib64 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -o $@ $< 

vanilladeriv.ll: vanilladeriv.cu | deps.jl
	$(CLANG) -L$(CUDA_PATH)/lib64 --cuda-path=$(CUDA_PATH) --cuda-gpu-arch=sm_70 -S -emit-llvm --cuda-device-only -o $@ $< 

vanilladeriv.ptx: vanilladeriv.ll | deps.jl
	$(LLC) -mcpu=sm_70 -o $@ $<

vanilladeriv.o: vanilladeriv.ptx
	ptxas -arch=sm_70 -lineinfo -v -warn-lmem-usage -warn-spills -o $@ $<

vanilladeriv.S: vanilladeriv.o
	nvdisasm -plr -g $< > $@


