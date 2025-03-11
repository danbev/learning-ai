info: src/info.cu
	nvcc -o $@ $<

wmma: src/wmma.cu
	# GeForce RTX 4070 has compute compatibility 8.6 (https://developer.nvidia.com/cuda-gpus)
	nvcc -arch=sm_89 -o $@ $<

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
	nvcc -lnppc -o $@ $<

matrix-mul-shared-mem: src/matrix-mul-shared-mem.cu
	nvcc -lnppc -o $@ $<

dump-array-add: array-add
	cuobjdump $<

array-add-ptx:
	nvcc -ptx src/array-add.cu
	@cat array-add.ptx


.PHONY: clean
clean:
	@${RM} threads inc hello-world.ptx info wmma streams graphs array-add
