# Make llama.cpp
llama: 
	cd llama.cpp && env GGML_DEBUG=1 LLAMA_DEBUG=1 DEBUG=1 make -j8

CXXFLAGS = -g -Wall -Illama.cpp/ -Illama.cpp/common

OBJS = llama.cpp/common.o \
       llama.cpp/ggml.o \
       llama.cpp/llama.o \
       llama.cpp/train.o \
       llama.cpp/ggml-alloc.o \
       llama.cpp/k_quants.o \
       llama.cpp/ggml-backend.o

simple-prompt: src/simple-prompt.cpp | llama
	$(CXX) $(CXXFLAGS) $^ -Lllama.cpp/ -o $@ $(OBJS)

clean:
	$(RM) -f simple-prompt
