info: src/info.cu
	nvcc -o $@ $<

wmma: src/wmma.cu
# GeForce RTX 4070 has compute compatibility 8.6 (https://developer.nvidia.com/cuda-gpus)
	nvcc -arch=sm_89 -o $@ $<

wmma-ptx:
	nvcc --ptx --gpu-architecture=sm_89 src/wmma.cu
	@cat wmma.ptx

hello-world-ptx: src/hello-world.cu
	nvcc -ptx $<
	@cat hello-world.ptx

threads: src/threads.cu
	nvcc -lnppc -o $@ $<

inc: src/inc.cu
	nvcc -lnppc -o $@ $<

array-add: src/array-add.cu
	nvcc -lnppc -o $@ $<

streams: src/streams.cu
	nvcc -lnppc -o $@ $<

graphs: src/graphs.cu
	nvcc -lnppc -o $@ $<

matrix-mul: src/matrix-mul.cu
	nvcc -lnppc -G -g -o $@ $<

matrix-mul-tiled: src/matrix-mul-tiled.cu
	nvcc -lnppc -G -g -o $@ $<

debug-matrix-tiled: matrix-mul-tiled
	cuda-gdb matrix-mul-tiled

dump-array-add: array-add
	cuobjdump $<

array-add-ptx:
	nvcc -ptx src/array-add.cu
	@cat array-add.ptx

template: src/template.cu
	nvcc -lnppc -o $@ $<

prefix-sum: src/prefix-sum.cu
	nvcc -arch=sm_89 -ptx $< -o $@.ptx
	nvcc -arch=sm_89 -cubin $< -o $@.cubin
	cuobjdump -sass $@.cubin > $@.sass
	nvcc -lnppc -G -g -o $@ $<

pagable: src/pagable.cu
	nvcc -lnppc -o $@ $<

run-pagable: pagable
	nsys profile --trace=cuda --cuda-memory-usage=true --force-overwrite=true -o $< $<
	nsys stats --report cuda_api_sum --force-overwrite true --force-export true $<.nsys-rep
	nsys analyze --rule cuda_memcpy_async --force-export=true $<.nsys-rep
	#nsys-ui $<.nsys-rep

.PHONY: clean
clean:
	@${RM} threads inc hello-world.ptx info wmma streams graphs array-add matrix-mul prefix-sum
