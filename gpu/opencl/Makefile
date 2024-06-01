main: src/main.c
	${CC} -v -DCL_TARGET_OPENCL_VERSION=300 $<  -o $@ -lOpenCL

dynamic-section: main
	@readelf -d main

ldd: main
	@ldd main


.PHONY: run
run: 
	#env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu LD_DEBUG=libs ./main
	env LD_DEBUG=libs ./main

.PHONY: clean
clean:
	${RM} main

.PHONY: tags
tags:
	ctags -R /urc/include/CL src/
