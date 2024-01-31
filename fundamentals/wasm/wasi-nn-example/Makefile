.PHONY: install-wasmedge
install-wasmedge:
	curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh \
	       	| bash -s -- --plugin wasi_nn-ggml
	echo "source $HOME/.wasmedge/env"

.PHONY: download-model
download-model:
	@mkdir -p models 
	curl -LO https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf \
		--output models/llama-2-7b-chat.Q5_K_M.gguf

build:
	cargo b --release --target wasm32-wasi

prompt = "What is LoRA?"
.PHONY: run-example
run-example:
	@env RUST_BACKTRACE=1 wasmedge --dir .:. --nn-preload llama-chat:GGML:AUTO:models/llama-2-7b-chat.Q5_K_M.gguf "target/wasm32-wasi/release/wasi-nn-example.wasm" llama-chat ${prompt}
