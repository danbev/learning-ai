
hello-world: src/hello-world.cu
	nvcc -allow-unsupported-compiler -o $@ $<

hello-world-ptx: src/hello-world.cu
	nvcc -ptx -allow-unsupported-compiler $<


.PHONY: clean
clean:
	${RM} hello-world
