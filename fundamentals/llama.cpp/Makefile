CUDA_DIR = /usr/local/cuda-12.2/lib64

# Make llama.cpp
llama: 
	cd llama.cpp && make -j8

llama-cuda: 
	source ./cuda-env.sh && cd llama.cpp && make -j8 LLAMA_CUBLAS=1

llama-vulkan:
	source ./cuda-env.sh && cd llama.cpp && make -j8 LLAMA_VULKAN=1

llama-kompute:
	cd llama.cpp && rm -rf build && mkdir build && cd build \
		&& cmake .. -DLLAMA_KOMPUTE=ON \
		&& make -j8

llama-cuda-debug:
	source ./cuda-env.sh && cd llama.cpp && make -j8 LLAMA_CUBLAS=1 LLAMA_DEBUG=1 DEBUG=1

# Use the following target with causion as it will probably mess up an
# actual run of a specific program. Just use it for debugging/learning and
# then switch back to the normal target for testing.
llama-debug: 
	cd llama.cpp && make -j8 LLAMA_DEBUG=1 DEBUG=1

CXXFLAGS = -g3 -Wall -Illama.cpp/include -Illama.cpp/ggml/include -Illama.cpp/common -Lllama.cpp/ -fopenmp

OBJS = llama.cpp/common.o \
       llama.cpp/build-info.o \
       llama.cpp/ggml.o \
       llama.cpp/llama.o \
       llama.cpp/train.o \
       llama.cpp/ggml-alloc.o \
       llama.cpp/ggml-quants.o \
       llama.cpp/ggml-backend.o \
       llama.cpp/unicode.o \
       llama.cpp/unicode-data.o \
       llama.cpp/sgemm.o \
       llama.cpp/sampling.o \
       llama.cpp/grammar-parser.o \
       llama.cpp/json-schema-to-grammar.o

kv-cache: src/kv-cache.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)

simple-prompt: src/simple-prompt.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)

run-simple-prompt: simple-prompt
	./simple-prompt

simple-prompt-cuda: OBJS += llama.cpp/ggml-cuda.o
simple-prompt-cuda: CXXFLAGS += -L${CUDA_DIR} -lcuda -lcublas -lculibos -lcudart -lcublasLt
simple-prompt-cuda: simple-prompt
run-simple-prompt-cuda: simple-prompt-vulkan
	./simple-prompt 0 33

simple-prompt-vulkan: OBJS += llama.cpp/ggml-vulkan.o
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
	git submodule update --recursive --remote --merge llama.cpp

.PHONY clean:
clean:
	$(RM) -f simple-prompt finetune

.PHONY clean-llama:
clean-llama:
	cd llama.cpp && make clean

quantize-llama-model:
	./llama.cpp/quantize models/llama-2-7b.gguf models/llama-2-7b-Q4.gguf Q4_1

download-llama-7b-q4:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf

download-llama-2-13b-chat.Q4:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf

download-llama-7b-q8:
	cd models && \
	wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q8_0.gguf

download-shakespeare:
	wget -P data/ https://raw.githubusercontent.com/brunoklein99/deep-learning-notes/master/shakespeare.txt

download-llama-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/meta-llama/Llama-2-7b

### open-llama targets
download-open_llama-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/openlm-research/open_llama_3b_v2

convert-open_llama-model:
	@python3.11 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert.txt && \
		python3.11 llama.cpp/convert.py open_llama_3b_v2 \
	        --outfile models/open-llama-2-7b.gguf --outtype f16

quantize-open-llama-model-q8:
	./llama.cpp/quantize models/open-llama-2-7b.gguf models/open-llama-2-7b-Q8_0.gguf Q8_0
	@ls -lh models/open-llama-2-7b-Q8_0.gguf

### llama-2-7b-chat targets
checkout-llama-2-7b-chat-hf-model:
	git clone --branch main --single-branch --depth 1 \
	https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/

convert-llama-2-7b-chat-hf-model:
	@python3.11 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert-hf-to-gguf.txt && \
		python3.11 llama.cpp/convert-hf-to-gguf.py Llama-2-7b-chat-hf \
	        --outfile models/llama-2-7b-hf-chat.gguf --outtype f16

convert-llama-2-7b-chat-model:
	@python3.11 -m venv venv
	@. venv/bin/activate && \
		pip install -r llama.cpp/requirements/requirements-convert.txt && \
		python3.11 llama.cpp/convert.py /home/danielbevenius/work/ai/llama/llama-2-7b-chat/ \
	        --outfile models/llama-2-7b-chat.gguf --outtype f16

quantize-llama-2-7b-chat-model-f16:
	./llama.cpp/quantize models/llama-2-7b-chat.gguf models/llama-2-7b-chat-f16.gguf F16

# I need to quantize this model to Q8_0.
quantize-llama-2-7b-chat-model-q8:
	./llama.cpp/quantize models/llama-2-7b-chat.gguf models/llama-2-7b-chat-Q8_0.gguf Q8_0
	@ls -lh models/llama-2-7b-chat-Q8_0.gguf

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
	$(CXX) $(CXXFLAGS) $^ -o $@ $(OBJS)
