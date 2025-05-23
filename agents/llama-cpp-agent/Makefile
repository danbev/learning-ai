### Echo Tool
echo-tool:
	cd tools/echo && cargo build --target wasm32-wasip1

echo-component: echo-tool | components
	wasm-tools component new tools/echo/target/wasm32-wasip1/debug/echo_tool.wasm \
    --adapt wit-lib/wasi_snapshot_preview1.reactor.wasm \
    -o components/echo-tool-component.wasm

components:
	mkdir components	

.PHONY: tool-runner
tool-runner:
	cd tool-runner && cargo build

run-echo-tool: tool-runner echo-component
	cd tool-runner && cargo run -- -c ../components/echo-tool-component.wasm --value "Hello"

inspect-echo-tool: echo-component
	wasm-tools component wit components/echo-tool-component.wasm

### Print Tool
print-tool:
	cd tools/print && cargo build --target wasm32-wasip1

print-component: print-tool | components
	wasm-tools component new tools/print/target/wasm32-wasip1/debug/print_tool.wasm \
    --adapt wit-lib/wasi_snapshot_preview1.reactor.wasm \
    -o components/print-tool-component.wasm

run-print-tool: tool-runner print-component
	cd tool-runner && cargo run -- -c ../components/print-tool-component.wasm --message "Something to print"

### Agent
run-agent-echo:
	cd agent && cargo run -- -m ../models/Phi-3-mini-4k-instruct-q4.gguf \
		-c ../components/echo-tool-component.wasm \
		-c ../components/print-tool-component.wasm \
		-p "Please echo the 'Hello'"

run-agent-print:
	cd agent && cargo run -- -m ../models/Phi-3-mini-4k-instruct-q4.gguf \
		-c ../components/echo-tool-component.wasm \
		-c ../components/print-tool-component.wasm \
		-p "Please print the following 'coffee'"

download-phi-mini-instruct: models
	cd models && \
	wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf

clean:
	cd tools/echo && cargo clean
	cd tool-runner && cargo clean

install-wasm32-wasip1:
	rustup target add wasm32-wasip1

.PHONY models:
models:
	mkdir -p models
