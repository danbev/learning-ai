CUDA_DIR = /usr/local/cuda-12.4/lib64
LLAMA_BUILD_DIR=llama.cpp/build

OS := $(shell uname -s)

# Make llama.cpp
llama-init:
ifeq ($(UNAME_S),Darwin)
	cd llama.cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_DEBUG=ON -DGGML_METAL_EMBED_LIBRARY=ON
else ifeq ($(UNAME_S),Linux)
	cd llama.cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_DEBUG=ON
else
	cd llama.cpp && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_DEBUG=ON
endif

llama: 
	cd llama.cpp && \
		cmake --build build -j8

llama-cuda: 
	cd llama.cpp && cmake -S . -B build -DLLAMA_CURL=ON \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DGGML_CPU_AARCH64=OFF \
    -DGGML_CUDA_F16=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Debug && cmake --build build -- -j 8

llama-vulkan:
	. ./cuda-env.sh && cd llama.cpp && \
		cmake --build build -DGGML_VULKAN=ON -j8

llama-kompute:
	cd llama.cpp && rm -rf build && mkdir build && \
		cmake --build build -DGGML_KOMPUTE=ON -j8

llama-cuda-debug:
	. ./cuda-env.sh && cd llama.cpp && \
		cmake -build build -DGGML_CUDA=ON -DLLAMA_DEBUG=ON -DCMAKE_BUILD_TYPE=DEBUG -DLLAMA_METAL=1 -j8

CXXFLAGS = -std=c++17 -g -Wall \
					 -Illama.cpp/include \
					 -Illama.cpp/ggml/include \
					 -Illama.cpp/common \
					 -Illama.cpp/src \
					 -Lllama.cpp/build/bin \
					 -lllama \
					 -lggml \
					 -lggml-cpu \
					 -Wl,-rpath,$(LLAMA_BUILD_DIR)/bin

ifeq ($(OS),Linux)
CXXFLAGS += -fopenmp
else ifeq ($(OS),Darwin)
CXXFLAGS += -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
endif

all: simple-prompt-multi

kv-cache: src/kv-cache.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJ)

simple-prompt: src/simple-prompt.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

simple-prompt-multi: src/simple-prompt-multi.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS)

batch: src/batch.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

embeddings: src/embeddings.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

logging: src/logging.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

logging-pre: src/logging.cpp
	$(CXX) $(CXXFLAGS) -E $^

simple-prompt-cuda: CXXFLAGS += -std=c++11 -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
simple-prompt-cuda: simple-prompt
run-simple-prompt-cuda: simple-prompt-cuda
	./simple-prompt 0 33

simple-prompt-vulkan: CXXFLAGS += -lvulkan
simple-prompt-vulkan: simple-prompt

run-simple-prompt-vulkan: simple-prompt-vulkan
	env GGML_VULKAN_DEVICE=1 ./simple-prompt 0 33

LDFLAGS=-Lllama.cpp/build/kompute/src -Lllama.cpp/build -Lllama.cpp/build -Lllama.cpp/build/kompute/src/logger

simple-prompt-kompute: src/simple-prompt.cpp
	$(CXX) -g3 -Wall -Illama.cpp/ -Illama.cpp/common $^ -o $@ \
	$(LDFLAGS) -lllama -lggml_static -lkompute -lvulkan -lkp_logger

monitor-gpu:
	@nvidia-smi -l

pre-simple-prompt: src/simple-prompt.cpp
	$(CXX) -E $(CXXFLAGS) $^  -o pre-simple-prompt.cpp

update-llama:
	git submodule update --recursive --remote llama.cpp

.PHONY clean:
clean:
	$(RM) -f simple-prompt simple-prompt-multi finetune tokenize

.PHONY clean-llama:
clean-llama:
	${RM} -rf llama.cpp/build

quantize-llama-model:
	./llama.cpp/llama-quantize models/llama-2-7b.gguf models/llama-2-7b-Q4.gguf Q4_1

download-llama-7b-q4: | models
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

download-jina: | models
	cd models && \
	wget https://huggingface.co/djuna/jina-embeddings-v2-small-en-Q5_K_M-GGUF/resolve/main/jina-embeddings-v2-small-en-q5_k_m.gguf

download-mixtral: | models
	cd models && \
	wget https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b-v0.1.Q2_K.gguf

.PHONY models:

models:
	mkdir -p models

download-llama-2-13b-chat.Q4:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf

download-llama-7b-q8:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf

download-llama-3-2-1B:
	cd models && \
	wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf


download-shakespeare:
	wget -P data/ https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

download-llama-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/meta-llama/Llama-2-7b

download-llama-3-model:
	cd .. && git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/meta-llama/Meta-Llama-3-8B

###### llama-2-7b-chat-hf targets  ######################
checkout-llama-2-7b-chat-hf-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/
	cd Llama-2-7b-chat-hf && git lfs install && git lfs pull

convert-llama-2-7b-chat-hf-model:
	@python3 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt && \
		python3 llama.cpp/convert_hf_to_gguf.py  Llama-2-7b-chat-hf \
	        --outfile models/llama-2-7b-hf-chat-f16.gguf --outtype f16

quantize-llama-2-7b-chat-hf-q4:
	./llama.cpp/llama-quantize models/llama-2-7b-hf-chat-f16.gguf models/llama-2-7b-hf-chat-q4.gguf Q4_K_S

#######################################################

###### llama-2-7b-chat targets  ######################
convert-llama-2-7b-chat-model:
	@python3 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt && \
		python3 llama.cpp/convert.py /home/danielbevenius/work/ai/llama/llama-2-7b-chat/ \
	        --outfile models/llama-2-7b-chat.gguf --outtype f16

quantize-llama-2-7b-chat-model-f16:
	./llama.cpp/llama-quantize models/llama-2-7b-chat.gguf models/llama-2-7b-chat-f16.gguf F16

# I need to quantize this model to Q8_0.
quantize-llama-2-7b-chat-model-q8:
	./llama.cpp/llama-quantize models/llama-2-7b-chat.gguf models/llama-2-7b-chat-Q8_0.gguf Q8_0
	@ls -lh models/llama-2-7b-chat-Q8_0.gguf

#######################################################

### rwkv targets
checkout-rwkv-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/RWKV/v6-Finch-1B6-HF/
	cd v6-Finch-1B6-HF && git lfs install && git lfs pull

convert-rwkv-model:
	@python3 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt && \
		python llama.cpp/convert_hf_to_gguf.py v6-Finch-1B6-HF \
	        --outfile models/v6-Finch-1B6-HF.gguf --outtype f16

run-rwkv-tokenize:
	./tokenize models/v6-Finch-1B6-HF.gguf

run-rwkv-simple-prompt:
	./simple-prompt 0 0 models/v6-Finch-1B6-HF.gguf

debug-rwkv-simple-prompt:
ifeq ($(OS),Linux)
	gdb --args ./simple-prompt 0 0 models/v6-Finch-1B6-HF.gguf
else
	lldb ./simple-prompt 0 0 models/v6-Finch-1B6-HF.gguf
endif

debug-rwkv-simple-prompt-multi:
ifeq ($(OS),Linux)
	gdb --args ./simple-prompt-multi 0 0 models/v6-Finch-1B6-HF.gguf
else
	lldb ./simple-prompt-multi 0 20 models/v6-Finch-1B6-HF.gguf
endif

inspect-rwkvf-model:
	@. venv/bin/activate && \
	llama.cpp/gguf-py/scripts/gguf_dump.py models/v6-Finch-1B6-HF.gguf && \
	deactivate

### Mamba targets
download-mamba-model:
	cd models && \
	wget https://huggingface.co/devingulliver/mamba-gguf/resolve/main/mamba-1.4b/mamba-1.4b-f16.gguf

inspect-mamba-model:
	@. venv/bin/activate && \
	llama.cpp/gguf-py/scripts/gguf_dump.py models/mamba-1.4b-f16.gguf && \
	deactivate

debug-mamba-simple-prompt-multi:
ifeq ($(OS),Linux)
	gdb --args ./simple-prompt-multi 0 0 models/mamba-1.4b-f16.gguf
else
	lldb ./simple-prompt-multi 0 20 models/mamba-1.4b-f16.gguf
endif

debug-mamba-simple-prompt:
ifeq ($(OS),Linux)
	gdb --args ./simple-prompt 0 0 models/mamba-1.4b-f16.gguf
else
	lldb ./simple-prompt 0 20 models/mamba-1.4b-f16.gguf
endif



### Finetune targets
finetune-llama-model: TYPE=llama
finetune-llama-model: TRAIN_DATA=data/assistent-training2.txt
finetune-llama-model: MODEL = models/llama-2-7b-chat.gguf
finetune-llama-model: CUDA_GPU_LAYERS = --n-gpu-layers 28
finetune-llama-model: SAMPLE_START = --sample-start '\#\#\#'
finetune-llama-model: EPOCHS = 10
finetune-llama-model: finetune-model

finetune-open-llama-model: TYPE=open-llama
finetune-open-llama-model: TRAIN_DATA = data/shakespeare.txt
finetune-open-llama-model: MODEL = models/open-llama-2-7b.gguf
finetune-open-llama-model: CUDA_GPU_LAYERS = --n-gpu-layers 28
finetune-open-llama-model: EPOCHS = 10
finetune-open-llama-model: finetune-model

finetune-model:
	llama.cpp/finetune \
        --model-base ${MODEL} \
        --checkpoint-in chk-${TYPE}-training-LATEST.gguf \
        --checkpoint-out chk-${TYPE}-training-ITERATION.gguf \
        --lora-out lora-${TYPE}-training-ITERATION.gguf \
        --train-data "${TRAIN_DATA}" \
        --save-every 10 \
        --threads 6 \
       	--adam-iter 30 \
        --batch 4 \
	--epochs ${EPOCHS} \
        --use-checkpointing \
        --ctx 78 \
        ${SAMPLE_START} \
        ${CUDA_GPU_LAYERS}

merge-llama-lora-adapter-with-base-model: TYPE=llama
merge-llama-lora-adapter-with-base-model: MODEL = models/llama-2-7b-chat.gguf
merge-llama-lora-adapter-with-base-model: merge-lora-adapter-with-base-model

merge-open-llama-lora-adapter-with-base-model: TYPE=open-llama
merge-open-llama-lora-adapter-with-base-model: MODEL = models/open-llama-2-7b.gguf
merge-open-llama-lora-adapter-with-base-model: merge-lora-adapter-with-base-model

merge-lora-adapter-with-base-model:
	./llama.cpp/export-lora \
        --model-base ${MODEL} \
	--model-out ${TYPE}-lora-merged-model.gguf \
	--lora lora-${TYPE}-training-LATEST.gguf

predict-lora:
	./llama.cpp/llama-cli -m models/open_llama-2-7b.gguf \
        --lora lora-training-LATEST.gguf \
	-n 100 \
        --n-gpu-layers 10 \
	-p "Love's fire heats water"

#predict-llama-lora-merged-model: MODEL = models/llama-2-7b-chat.gguf
predict-llama-lora-merged-model: MODEL = llama-lora-merged-model.gguf
predict-llama-lora-merged-model: LAYERS = 27
predict-llama-lora-merged-model: PROMPT = "<s>[INST] Can you show me a summary of RHSA-2024:0102? [/INST]"
predict-llama-lora-merged-model: predict-lora-merged-model

predict-open-llama-lora-merged-model: MODEL = open-llama-lora-merged-model.gguf
predict-open-llama-lora-merged-model: LAYERS = 27
predict-open-llama-lora-merged-model: PROMPT = "Love's fire heats water"
predict-open-llama-lora-merged-model: predict-lora-merged-model

predict-lora-merged-model:
	./llama.cpp/llama-cli -m ${MODEL} \
        -n 100 \
        --n-gpu-layers ${LAYERS} \
        --no-display-prompt \
        --log-disable \
        --threads 6 \
        --ctx-size 512 \
        -p ${PROMPT}

#--lora lora-llama-training-LATEST.gguf \
#--lora-scaled lora-llama-training-LATEST.gguf 0.1 \
#-p "<s>[INST] Can you show me a summary of RHSA-2024:0088? [/INST]"
#-p "<s>[INST] What is the capital of Sweden? [/INST]"
#-p "<s>[INST] Can you show me a summary of 2 things to do Stockholm? [/INST]"

llama-cli:
	cd llama.cpp && make llama-cli LLAMA_CUBLAS=1

check-llama.cpp-finetune-include:
	/home/danielbevenius/work/c++/include-what-you-use/build/bin/include-what-you-use -Illama.cpp -I/usr/lib/clang/17/include/ -Illama.cpp/common llama.cpp/examples/finetune/finetune.cpp

check-finetune-include:
	/home/danielbevenius/work/c++/include-what-you-use/build/bin/include-what-you-use -Illama.cpp -I/usr/lib/clang/17/include/ -Illama.cpp/common src/finetune.cpp

.PHONY clean-lora-files:
clean-lora-files:
	${RM} -f lora-llama-training-*.gguf
	${RM} -f lora-open-llama-training-*.gguf
	${RM} -f chk-llama-training-*.gguf
	${RM} -f chk-open-llama-training-*.gguf

tokenize-file: CXXFLAGS += -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
tokenize-file: src/tokenize_file.cc
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJ)

tokenize: src/tokenize.cpp
	#$(CXX) $(CXXFLAGS) -framework Metal -framework Foundation -framework MetalKit -framework Accelerate $^ -o $@ $(OBJ) $(OBJ) $(OBJ_COMMON) $(LIBS)
	echo $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJ) $(OBJ_GGML) $(OBJ_COMMON) $(LIBS)

simple-dyn-link: LDFLAGS = -Lllama.cpp/build/src -Lllama.cpp/build/ggml/src -lllama -lggml
simple-dyn-link: llama.cpp/examples/simple/simple.cpp
	echo "Compiling..."
	cd llama.cpp && cmake -S . -B build -DLLAMA_CURL=ON && cmake --build build
	$(CXX) $(CXXFLAGS) llama.cpp/examples/simple/simple.cpp -o $@ -Lllama.cpp/build/src -Lllama.cpp/build/ggml/src -lllama -lggml

ifeq ($(OS),Darwin)
LD_VARNAME := DYLD_LIBRARY_PATH
MODEL_PATH := models/llama-2-7b.Q4_0.gguf
else
LD_VARNAME := LD_LIBRARY_PATH
MODEL_PATH := models/llama-2-7b.Q4_K_M.gguf
endif

run-simple-dyn-link:
	env ${LD_VARNAME}=llama.cpp/build/src:llama.cpp/build/ggml/src \
	./simple-dyn-link -m ${MODEL_PATH} -n 10 -ngl 33 "What is LoRA?"

debug-simple-dyn-link:
ifeq ($(OS),Darwin)
	lldb -o 'settings set target.env-vars DYLD_LIBRARY_PATH=llama.cpp/build/src:llama.cpp/build/ggml/src' \
	./simple-dyn-link -- -m ${MODEL_PATH} -n 10 -ngl 33 "What is LoRA?"
else
	env ${LD_VARNAME}=llama.cpp/build/src:llama.cpp/build/ggml/src \
	gdb --args ./simple-dyn-link -- -m ${MODEL_PATH} -n 10 -ngl 33 "What is LoRA?"
endif
