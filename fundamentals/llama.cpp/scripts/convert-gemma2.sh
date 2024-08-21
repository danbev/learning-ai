#!/bin/bash
source venv/bin/activate

model_name=gemma-2-9b-it
model_dir=~/.cache/huggingface/hub/models--google--${model_name}/snapshots/93be03fbe3787f19bf03a4b1d3d75d36cb1f6ace

#python convert-hf-to-gguf.py $model_dir --outfile=models/${model_name}.gguf --outtype f16 

python convert-hf-to-gguf.py $model_dir --outfile=models/${model_name}.gguf --outtype f16 --split-max-tensors 100

deactivate
