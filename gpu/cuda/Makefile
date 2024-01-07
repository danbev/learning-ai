
hello-world: src/hello-world.cu
	nvcc -allow-unsupported-compiler -o $@ $<


.PHONY: clean
clean:
	${RM} hello-world
