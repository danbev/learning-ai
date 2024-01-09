
hello-world: src/hello-world.cu
	nvcc -lnppc -o $@ $<

hello-world-ptx: src/hello-world.cu
	nvcc -ptx $<
	@cat hello-world.ptx

threads: src/threads.cu
	nvcc -lnppc -o $@ $<


.PHONY: clean
clean:
	${RM} hello-world threads
