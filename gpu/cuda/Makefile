
hello-world: src/hello-world.cu
	nvcc -lnppc -o $@ $<

hello-world-ptx: src/hello-world.cu
	nvcc -ptx $<
	@cat hello-world.ptx


.PHONY: clean
clean:
	${RM} hello-world
