#!/bin/bash
#
set -e

llama_cpp_dir=/home/danbev/work/llama.cpp
model_dir=models/Qwen2.5-0.5B-Instruct-gptq-4bit
display_name="Qwen2.5-0.5B-Instruct-gptq-4bit"

python $llama_cpp_dir/convert_hf_to_gguf.py ${model_dir} \
	--outtype bf16 \
	--model-name ${display_name} \
	--outfile converted/${display_name}.gguf
