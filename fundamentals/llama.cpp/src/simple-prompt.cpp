#include "common.h"
#include "llama.h"

#include <iostream>

int main(int argc, char** argv) {
    std::cout << "llama.cpp example" << std::endl;
    gpt_params params;
    params.model = "models/llama-2-13b-chat.Q4_0.gguf";
    std::cout << "params.n_threads: " << params.n_threads << std::endl;

    llama_backend_init(params.numa);
    llama_model_params model_params = llama_model_default_params();

    llama_model* model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, params.model.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    std::vector<llama_token> input_tokens;
    std::string query = "What is LoRA?";
    std::cout << "query: " << query << std::endl;

    input_tokens = ::llama_tokenize(ctx, query, true);
    std::cout << "input_tokens: " << std::endl;
    for (auto token : input_tokens) {
        std::cout << token << " :" << llama_token_to_piece(ctx, token) << std::endl;
    }

    llama_batch batch = llama_batch_init(512, 0);
    batch.n_tokens = input_tokens.size();
    std::cout << "batch.n_tokens: " << batch.n_tokens << std::endl;

    // Add the tokens to the batch
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i] = input_tokens[i];
        batch.pos[i] = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
    }
    // Instruct llama to generate the logits for the last token
    batch.logits[batch.n_tokens - 1] = true;

    // What does decode do?
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }
    std::cout << "logits: " << batch.logits[batch.n_tokens-1] << std::endl;

    return 0;
}
