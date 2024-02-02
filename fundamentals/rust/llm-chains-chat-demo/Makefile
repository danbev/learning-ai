.PHONY: download-model
download-model:
	@mkdir -p models 
	@cd models && curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/5db6994dca7288297c59693b5e27b62f22a54e1f/llama-2-7b-chat.Q4_0.gguf
		
.PHONY: start-quandrant-server
start-quandrant-server:
	cd ../../../vector-databases/qdrant/rust && \
	make start-qdrant-server 

build-llama-cuda: src/main-llama.rs
	env LLM_CHAIN_CUDA_LIB_PATH=/usr/local/cuda-12.2/lib64 LLM_CHAIN_CUDA=true cargo b --bin llama -vv

run-llama-cuda: src/main-llama.rs
	env LLM_CHAIN_CUDA_LIB_PATH=/usr/local/cuda-12.2/lib64 LLM_CHAIN_CUDA=true cargo r --bin llama -vv
