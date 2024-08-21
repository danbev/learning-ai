# Testing Self-Extend
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 --grp-attn-n 4 --grp-attn-w 32 -f pg1184.txt -c 16384 --temp 0
# llama-2-7b.Q4_0.gguf was trained on a 4096 context size
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 100 --grp-attn-n 4 --grp-attn-w 32
#gdb --args ./llama-cli -m /home/danbev/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf -ngl 10 -f self-extend.txt -c 4096 --temp 1 -n 100 --grp-attn-n 4 --grp-attn-w 512
#gdb --args ./llama-cli -m /home/danbev/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf -ngl 10 -f self-extend.txt -c 8000 --temp 1 -n 200 --grp-attn-n 4 --grp-attn-w 128
#gdb --args ./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8000 --temp 1 -n 200 --grp-attn-n 2 --grp-attn-w 2048
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8000 --temp 1 -n 200 --grp-attn-n 128 --grp-attn-w 2048
gdb --args ./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8000 --temp 1 --grp-attn-n 2 --grp-attn-w 2048
#./llama-cli -m /home/danbev/.cache/lm-studio/models/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/tinyllama-1.1b-1t-openorca.Q2_K.gguf -ngl 10 -f self-extend.txt -c 5000 --temp 1 -n 100 --grp-attn-n 4 --grp-attn-w 32
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256
