#!/bin/bash

set -e

#export CUDA_LAUNCH_BLOCKING=1
#export LLAMA_KV_CACHE_DEBUG=1

llama_dir=/home/danbev/work/llama.cpp

build_dir=${llama_dir}/build-dgx-spark-debug
model=${PWD}/converted/Qwen2.5-0.5B-Instruct-nvfp4.gguf
cmd=llama-completion

pushd $llama_dir

cmake --build ${build_dir} --target ${cmd}

${build_dir}/bin/${cmd} -m ${model} \
	--prompt "What is the capital of Sweden?" \
	--verbose-prompt \
	-n 20 --jinja -ngl 99 --no-warmup -no-cnv

popd
