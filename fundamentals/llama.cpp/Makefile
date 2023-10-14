# Make llama.cpp
llama: 
	cd llama.cpp && make -j8 LLAMA_DEBUG=1 DEBUG=1

CXXFLAGS = -g3 -Wall -Illama.cpp/ -Illama.cpp/common -Lllama.cpp/

OBJS = llama.cpp/common.o \
       llama.cpp/ggml.o \
       llama.cpp/llama.o \
       llama.cpp/train.o \
       llama.cpp/ggml-alloc.o \
       llama.cpp/k_quants.o \
       llama.cpp/ggml-backend.o

simple-prompt: src/simple-prompt.cpp | llama
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)

pre-simple-prompt: src/simple-prompt.cpp
	$(CXX) -E $(CXXFLAGS) $^  -o pre-simple-prompt.cpp

clean:
	$(RM) -f simple-prompt
