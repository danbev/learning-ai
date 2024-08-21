#./llama-embedding -m models/llama-2-7b-chat.Q4_K_M.gguf --no-warmup --pooling mean  -p "What is LoRA?"
#gdb --args ./llama-embedding -m models/llama-2-7b-chat.Q4_K_M.gguf --no-warmup --pooling mean  -p "What is LoRA?"
#gdb --args ./llama-embedding -m models/llama-2-7b-chat.Q4_K_M.gguf --no-warmup --pooling last  -p "What is LoRA?"
gdb --args ./llama-embedding -m models/llama-2-7b-chat.Q4_K_M.gguf --no-warmup --pooling cls  -p "What is LoRA?"
