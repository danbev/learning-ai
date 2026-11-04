#!/bin/bash
#
set -e

llama_cpp_dir=/home/danbev/work/llama.cpp
qformat=${1:-nvfp4}
display_name="Qwen2.5-0.5B-Instruct-${qformat}"
model_dir=models/${display_name}/

python $llama_cpp_dir/convert_hf_to_gguf.py ${model_dir} \
	--model-name ${display_name} \
	--outfile converted/${display_name}.gguf
