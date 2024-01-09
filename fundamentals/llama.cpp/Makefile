CUDA_DIR = /usr/local/cuda-12.2/lib64

# Make llama.cpp
llama: 
	cd llama.cpp && make -j8

llama-cuda: 
	source ./cuda-env.sh && cd llama.cpp && make -j8 LLAMA_CUBLAS=1

# Use the following target with causion as it will probably mess up an
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
       llama.cpp/ggml-backend.o \
       llama.cpp/ggml-cuda.o 

simple-prompt: src/simple-prompt.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)

simple-prompt-cuda: CXXFLAGS += -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
simple-prompt-cuda: simple-prompt

monitor-gpu:
	@nvidia-smi -l

pre-simple-prompt: src/simple-prompt.cpp
	$(CXX) -E $(CXXFLAGS) $^  -o pre-simple-prompt.cpp

finetune: src/finetune.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)

finetune-cuda: CXXFLAGS += -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
finetune-cuda: finetune

update-llama:
	git submodule update --remote --merge llama.cpp

clean:
	$(RM) -f simple-prompt finetune

clean-llama:
	cd llama.cpp && make clean

download-falcon-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/tiiuae/falcon-7b

# This target requires the falcon model to be downloaded first.
convert-falcon-model:
	@python3.11 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements-hf-to-gguf.txt && \
		python3.11 llama.cpp/convert-hf-to-gguf.py falcon-7b \
	        --outfile models/falcon-7b.gguf --outtype f16

quantize-falcon-model:
	./llama.cpp/quantize models/falcon-7b.gguf models/falcon-7b.Q4_1.gguf Q4_1


download-llama-7b-q4:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf

download-llama-7b-q8:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf

download-shakespeare:
	wget -P data/ https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

finetune-llama-2-7b.Q4_0-model:
	./finetune \
        --model-base models/llama-2-7b.Q4_0.gguf \
        --checkpoint-in chk-llama-2-7b.Q4_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-llama-2-7b.Q4_0-shakespeare-ITERATION.gguf \
        --lora-out lora-llama-2-7b.Q4_0-shakespeare-ITERATION.bin \
        --train-data "data/shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing

finetune-llama-2-7b.Q4_0-model-debug:
	gdb --args ./finetune \
        --model-base models/llama-2-7b.Q4_0.gguf \
        --checkpoint-in chk-llama-2-7b.Q4_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-llama-2-7b.Q4_0-shakespeare-ITERATION.gguf \
        --lora-out lora-llama-2-7b.Q4_0-shakespeare-ITERATION.bin \
        --train-data "data/shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing

llama-main:
	cd llama.cpp && make main

finetune-predict: llama-main
	./llama.cpp/main -m models/llama-2-7b.Q4_0.gguf \
        --lora lora-llama-2-7b.Q4_0-shakespeare-LATEST.bin \
	-p "Love's fire heats water"

check-llama.cpp-finetune-include:
	/home/danielbevenius/work/c++/include-what-you-use/build/bin/include-what-you-use -Illama.cpp -I/usr/lib/clang/17/include/ -Illama.cpp/common llama.cpp/examples/finetune/finetune.cpp

check-finetune-include:
	/home/danielbevenius/work/c++/include-what-you-use/build/bin/include-what-you-use -Illama.cpp -I/usr/lib/clang/17/include/ -Illama.cpp/common src/finetune.cpp
