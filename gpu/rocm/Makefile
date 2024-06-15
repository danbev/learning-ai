
helloworld: src/helloworld.cpp
	# use --keep to inspect the generated assembly code
	env HIP_PLATFORM=nvidia hipcc -o $@ $<

.PHONY: clean
clean:
	${RM} helloworld
