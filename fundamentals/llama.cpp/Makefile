# Make llama.cpp
llama: 
	cd llama.cpp && make -j8

# Use the following target with causion as it will probably mess up and
# actual run of a specific program. Just use it for debugging/learning and
# then switch back to the normal target for testing.
llama-debug: 
	cd llama.cpp && make -j8 LLAMA_DEBUG=1 DEBUG=1

CXXFLAGS = -g3 -Wall -Illama.cpp/ -Illama.cpp/common -Lllama.cpp/

OBJS = llama.cpp/common.o \
       llama.cpp/build-info.o \
       llama.cpp/ggml.o \
       llama.cpp/llama.o \
       llama.cpp/train.o \
       llama.cpp/ggml-alloc.o \
       llama.cpp/ggml-quants.o \
       llama.cpp/ggml-backend.o

simple-prompt: src/simple-prompt.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)

pre-simple-prompt: src/simple-prompt.cpp
	$(CXX) -E $(CXXFLAGS) $^  -o pre-simple-prompt.cpp

update-llama:
	git submodule update --remote --merge

clean:
	$(RM) -f simple-prompt

clean-llama:
	cd llama.cpp && make clean
