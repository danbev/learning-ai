NVCC ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?=-G -g

# Use the local GPU by default; override with e.g. CUDA_ARCH=sm_89.
CUDA_FLAGS = $(NVCCFLAGS) -arch=$(CUDA_ARCH)

info: src/info.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

wmma: src/wmma.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

wmma-ptx: src/wmma.cu
	$(NVCC) $(CUDA_FLAGS) --ptx src/wmma.cu
	@cat wmma.ptx

hello-world-ptx: src/hello-world.cu
	$(NVCC) $(CUDA_FLAGS) -ptx $<
	@cat hello-world.ptx

threads: src/threads.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

inc: src/inc.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

array-add: src/array-add.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

streams: src/streams.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

graphs: src/graphs.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

shuffle: src/shuffle.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

shuffle-sass: src/shuffle.cu
	$(NVCC) -arch=$(CUDA_ARCH) -cubin $< -o shuffle.cubin
	cuobjdump -sass shuffle.cubin | grep -E "SHFL|IADD|FUNC"

matrix-mul: src/matrix-mul.cu
	$(NVCC) $(CUDA_FLAGS) -G -g -o $@ $<

matrix-mul-tiled: src/matrix-mul-tiled.cu
	$(NVCC) $(CUDA_FLAGS) -G -g -o $@ $<

debug-matrix-tiled: matrix-mul-tiled
	cuda-gdb matrix-mul-tiled

dump-array-add: array-add
	cuobjdump $<

array-add-ptx: src/array-add.cu
	$(NVCC) $(CUDA_FLAGS) -ptx src/array-add.cu
	@cat array-add.ptx

template: src/template.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

prefix-sum: src/prefix-sum.cu
	$(NVCC) $(CUDA_FLAGS) -ptx $< -o $@.ptx
	$(NVCC) $(CUDA_FLAGS) -cubin $< -o $@.cubin
	cuobjdump -sass $@.cubin > $@.sass
	$(NVCC) $(CUDA_FLAGS) -G -g -o $@ $<

pagable: src/pagable.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

run-pagable: pagable
	nsys profile --trace=cuda --cuda-memory-usage=true --force-overwrite=true -o $< $<
	nsys stats --report cuda_api_sum --force-overwrite true --force-export true $<.nsys-rep
	nsys analyze --rule cuda_memcpy_async --force-export=true $<.nsys-rep
	#nsys-ui $<.nsys-rep

pinned: src/pinned.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

ldmatrix: src/ldmatrix.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< -Xptxas -v

dp4a: src/dp4a.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< -Xptxas -v

cp_async: src/cp_async.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

fastmod: src/fastmod.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

misalign: src/misalign.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

nvfp4: src/nvfp4.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $< -Xptxas -v

nvfp4-api: src/nvfp4-api.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

run-pinned: pinned
	nsys profile --trace=cuda --cuda-memory-usage=true --force-overwrite=true -o $< $<
	#nsys stats --report cuda_api_sum --force-overwrite true --force-export true $<.nsys-rep
	nsys analyze --rule cuda_memcpy_async --force-export=true $<.nsys-rep

vmm: src/vmm.cu
	$(NVCC) $(CUDA_FLAGS) -lcuda -G -g -o $@ $<

mmvq: src/mmvq.cu
	$(NVCC) $(CUDA_FLAGS) -o $@ $<

EXAMPLES = info wmma threads inc array-add streams graphs matrix-mul \
           matrix-mul-tiled template prefix-sum pagable pinned ldmatrix dp4a \
           cp_async fastmod misalign vmm mmvq

.PHONY: all clean wmma-ptx array-add-ptx
all: $(EXAMPLES)

clean:
	$(RM) $(EXAMPLES) hello-world.ptx wmma.ptx array-add.ptx prefix-sum.ptx prefix-sum.cubin prefix-sum.sass
