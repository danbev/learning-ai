#include "llama.h"
#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>

std::vector<llama_token> tokenize_prompt(llama_model* model, std::string prompt) {
    const int add_bos_token = llama_add_bos_token(model);
    const bool add_bos  = add_bos_token != -1 ? bool(add_bos_token) :
        (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM); // SPM = SentencePiece Model

    int n_tokens = prompt.length() + add_bos;
    std::vector<llama_token> input_tokens(n_tokens);
    n_tokens = llama_tokenize(model,
                              prompt.data(),
                              prompt.length(),
                              input_tokens.data(),
                              input_tokens.size(),
                              true,
                              false);
    if (n_tokens < 0) {
        input_tokens.resize(-n_tokens);
        llama_tokenize(model,
                prompt.data(),
                prompt.length(),
                input_tokens.data(),
                input_tokens.size(), add_bos, false);
    } else {
        input_tokens.resize(n_tokens);
    }
    return input_tokens;
}

llama_batch create_batch(int size, std::vector<std::vector<llama_token>> input_tokens) {
    int n_prompts = input_tokens.size();
    printf("Creating new llama_batch with %d sequences\n", n_prompts);

    // Create a single batch for both prompts.
    llama_batch batch = llama_batch_init(size, /*embd*/ 0, /*n_seq_max*/ n_prompts);

    for (size_t p = 0; p < input_tokens.size(); p++) {
        printf("Processing prompt %ld, size = %ld, batch_n_tokens: %d \n", p,input_tokens.size(), batch.n_tokens);
        std::vector<llama_token> prompt_tokens = input_tokens[p];
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            int idx = batch.n_tokens;
            batch.token[idx] = prompt_tokens[i];
            batch.pos[idx] = i,
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = p;  // the sequence id
            batch.logits[idx] = i == prompt_tokens.size() - 1;

            batch.n_tokens++;
            printf("idx: %4d, token: %6d, seq_id: %ld, logits: %d\n", idx, prompt_tokens[i], p, batch.logits[idx]);
        }
    }
    return batch;
}

void print_batch(llama_batch batch) {
    fprintf(stderr, "batch.n_tokens: %d\n", batch.n_tokens);
    fprintf(stderr, "batch.tokens: [");
    for (int i = 0; i < batch.n_tokens; i++) {
        fprintf(stderr, "%d, ", batch.token[i]);
    }
    fprintf(stderr, "]\n");
}

int main(int argc, char** argv) {
    fprintf(stdout, "llama.cpp batch exploration\n");
    llama_model_params model_params = llama_model_default_params();
    std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;

    std::string prompt1 = "What is LoRA?";
    std::string prompt2 = "Dan loves ice cream";

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    ctx_params.n_seq_max = 6;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    fprintf(stderr, "prompt1: %s\n", prompt1.c_str());
    fprintf(stderr, "prompt2: %s\n", prompt2.c_str());

    // Tokenize the prompts.
    std::vector<llama_token> input_tokens1 = tokenize_prompt(model, prompt1);
    std::vector<llama_token> input_tokens2 = tokenize_prompt(model, prompt2);

    llama_batch batch = create_batch(512, {input_tokens1, input_tokens2});
    print_batch(batch);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    float* logits = llama_get_logits_ith(ctx, -1);
    int embd_size = llama_n_embd(model);
    for (int i = embd_size - 10; i < embd_size; i++) {
        fprintf(stderr, "logits[%d]: %f\n", i, logits[i]);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
