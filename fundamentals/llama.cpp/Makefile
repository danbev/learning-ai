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


quantize-llama-model:
	./llama.cpp/quantize models/llama-2-7b.gguf models/llama-2-7b-Q4.gguf Q4_1

download-llama-7b-q4:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf

download-llama-7b-q8:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf

download-shakespeare:
	wget -P data/ https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

download-llama-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/meta-llama/Llama-2-7b

download-open_llama-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/openlm-research/open_llama_3b_v2

# This target requires the model to be downloaded first.
convert-open_llama-model:
	@python3.11 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert.txt && \
		python3.11 llama.cpp/convert.py open_llama_3b_v2 \
	        --outfile models/open_llama-2-7b.gguf --outtype f16

finetune-open_llama-model-cuda:
	./finetune \
        --model-base models/open_llama-2-7b.gguf \
        --checkpoint-in chk-open_llama-shakespeare-LATEST.gguf \
        --checkpoint-out chk-open_llama-shakespeare-ITERATION.gguf \
        --lora-out lora-open_llama-shakespeare-ITERATION.gguf \
        --train-data "data/shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing \
        --n-gpu-layers 27

merge-lora-adapter-with-base-model:
	./llama.cpp/export-lora \
        --model-base models/open_llama-2-7b.gguf \
	--model-out lora-merged-model.gguf \
	--lora lora-open_llama-shakespeare-LATEST.gguf

finetune-open_llama-model-debug:
	gdb --args ./finetune \
        --model-base models/open_llama-2-7b.gguf \
        --checkpoint-in chk-open_llama-shakespeare-LATEST.gguf \
        --checkpoint-out chk-open_llama-shakespeare-ITERATION.gguf \
        --lora-out lora-open_llama-shakespeare-ITERATION.gguf \
        --train-data "data/shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing

finetune-predict: llama-main
	./llama.cpp/main -m models/open_llama-2-7b.gguf \
	-n 30 \
        --n-gpu-layers 27 \
	-p "Love's fire heats water"

finetune-predict-lora: llama-main
	./llama.cpp/main -m models/open_llama-2-7b.gguf \
        --lora lora-open_llama-shakespeare-LATEST.gguf \
	-n 30 \
        --n-gpu-layers 10 \
	-p "Love's fire heats water"

finetune-predict-lora-merged-model: llama-main
	./llama.cpp/main -m lora-merged-model.gguf \
	-n 30 \
        --n-gpu-layers 27 \
	-p "Love's fire heats water"

llama-main:
	cd llama.cpp && make main LLAMA_CUBLAS=1


check-llama.cpp-finetune-include:
	/home/danielbevenius/work/c++/include-what-you-use/build/bin/include-what-you-use -Illama.cpp -I/usr/lib/clang/17/include/ -Illama.cpp/common llama.cpp/examples/finetune/finetune.cpp

check-finetune-include:
	/home/danielbevenius/work/c++/include-what-you-use/build/bin/include-what-you-use -Illama.cpp -I/usr/lib/clang/17/include/ -Illama.cpp/common src/finetune.cpp
