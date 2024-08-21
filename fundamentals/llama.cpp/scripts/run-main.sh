#./llama-cli -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt
#gdb --args ./llama-cli -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt


## Run with session file
#gdb --args ./llama-cli -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt --prompt-cache main-session.txt
#./llama-cli -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt --prompt-cache main-session.txt
#gdb --args ./llama-cli -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'Hello world' -n 3 --verbose-prompt --prompt-cache main-session.txt --dump-kv-cache


#gdb --args ./llama-cli -m models/tinyllama-1.1b-1t-openorca.Q2_K.gguf --prompt 'Hello world' -n 3 --verbose-prompt --dump-kv-cache

#gdb --args ./llama-cli -m /home/danbev/.cache/lm-studio/models/TheBloke/phi-2-GGUF/phi-2.Q3_K_M.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt --temp 0
#./llama-cli -m /home/danbev/.cache/lm-studio/models/TheBloke/phi-2-GGUF/phi-2.Q3_K_M.gguf --prompt 'The answer to 1 + 1 is' -n 5 --verbose-prompt --temp 0 --top-p 0.950 --min-p 0.05 --repeat-penalty 1.1  --typical 1


#./llama-cli -m /home/danbev/.cache/lm-studio/models/TheBloke/phi-2-GGUF/phi-2.Q3_K_M.gguf --color -i -r "User:" -f prompts/chat-with-bob.txt

#./llama-cli -m models/gemma-2-9b.gguf -p "<start_of_turn>user
#What is LoRA?<end_of_turn>
#<start_of_turn>model"

#./llama-cli -m models/gemma-2-9b.gguf -ngl 15 -p "Hi"
#./llama-cli -m ~/.cache/lm-studio/models/bartowski/gemma-2-9b-it-GGUF/gemma-2-9b-it-Q4_K_S.gguf -ngl 15 -p "<start_of_turn>user\nWhat is LoRA?<end_of_turn>\n<start_of_turn>model"
#./llama-cli -m models/gemma-2-9b-it.gguf -ngl 15 -p "<start_of_turn>user\nWhat is LoRA?<end_of_turn>\n<start_of_turn>model"
#./llama-cli -m models/gemma-2-9b-it.gguf -ngl 15 -p "What is LoRA?"
#./llama-cli -m models/gemma-2-9b-it.gguf -ngl 15 -p "Dan loves icecream"
#./llama-cli -m models/gemma-2-9b-it.gguf -dkvc -ngl 15 -p "Dan loves icecream"
#gdb --args ./llama-cli -m models/gemma-2-9b-it.gguf --grp-attn-n 2 --grp-attn-w 4 -p "Dan loves icecream"


#gdb --args ./llama-cli -m models/llama-2-7b.Q4_0.gguf --no-warmup --rope-scaling yarn --rope-freq-scale 1 --yarn-ext-factor 1.0 -ngl 10 -p "What is LoRA?" -n 10
#./llama-cli -m models/llama-2-7b.Q4_0.gguf --no-warmup --rope-scaling yarn --rope-freq-scale 1 --yarn-ext-factor 1.0 -ngl 10 -p "What is LoRA?" -n 10
# Testing Self-Extend
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 --grp-attn-n 4 --grp-attn-w 32 -f pg1184.txt -c 16384 --temp 0
# llama-2-7b.Q4_0.gguf was trained on a 4096 context size
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256 --grp-attn-n 4 --grp-attn-w 256
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256
#./llama-cli -m models/llama-2-7b.Q4_0.gguf -ngl 10 -f self-extend.txt -c 8192 --temp 0 -n 256

./llama-cli -m models/mamba-gpt-7b-q4_0.gguf --no-warmup -ngl 10 -p "What is LoRA?" -n 10
