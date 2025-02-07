info: src/info.cu
	nvcc -o $@ $<

wmma: src/wmma.cu
	# GeForce RTX 4080 has compute compatibility 8.6 (https://developer.nvidia.com/cuda-gpus)
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

dump-array-add: array-add
	cuobjdump $<


.PHONY: clean
clean:
	${RM} hello-world threads inc hello-world.ptx minimal wmma streams graphs array-add
