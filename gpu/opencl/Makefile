main: src/main.c
	${CC} -DCL_TARGET_OPENCL_VERSION=300 $<  -o $@ -lOpenCL

.PHONY: run
run: 
	env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./main

.PHONY: clean
clean:
	${RM} main
