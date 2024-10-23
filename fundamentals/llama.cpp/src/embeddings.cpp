#include "llama.h"
#include "llama-sampling.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>

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

llama_batch create_batch(int size, std::vector<llama_token> input_tokens, llama_model* model) {
    llama_batch batch = llama_batch_init(size, 0, 1);
    for (size_t i = 0; i < input_tokens.size(); i++) {
        int token_id = input_tokens[i];
        batch.token[i] = token_id;
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
        batch.n_tokens++;
    }
    return batch;
}

std::string token_as_string(llama_model* model, llama_token token) {
    int lsplit = 0;
    bool special = false;
    std::vector<char> piece(8, 0);
    int n_tokens = llama_token_to_piece(model, token, piece.data(), piece.size(), lsplit, special);
    if (n_tokens < 0) {
        piece.resize(-n_tokens);
        llama_token_to_piece(model, token, piece.data(), piece.size(), lsplit, special);
    } else {
        piece.resize(n_tokens);
    }
    return std::string(piece.data(), piece.size());
}

const char* RED = "\033[0;31m";
const char* GREEN = "\033[0;32m";
const char* BLUE = "\033[0;34m";
const char* ORANGE = "\033[0;33m";  // Actually yellow, but often appears as orange in many terminals
const char* RESET = "\033[0m";

int main(int argc, char** argv) {
    fprintf(stdout, "llama.cpp embedding exploration\n");
    llama_model_params model_params = llama_model_default_params();
    std::string model_path = "models/llama-2-7b-chat.Q4_K_M.gguf";

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;

    std::string prompt = "What is LoRA?";

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
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    ctx_params.embeddings = true;

    llama_context* embd_ctx = llama_new_context_with_model(model, ctx_params);
    if (embd_ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("%sprompt: %s%s\n", GREEN, prompt.c_str(), RESET);

    std::vector<llama_token> input_tokens = tokenize_prompt(model, prompt);

    llama_batch batch = create_batch(ctx_params.n_batch, input_tokens, model);

    if (llama_decode(embd_ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    float* embd = nullptr;
    embd = llama_get_embeddings_seq(embd_ctx, 0);
    printf("%sGenerated embeddings for prompt. Embedding size: %d.\%s\n", BLUE, llama_n_embd(model), RESET);

    ctx_params.embeddings = false;
    llama_context* inf_ctx = llama_new_context_with_model(model, ctx_params);
    // So now that we have the embeddings, lets try passing them to the LLM
    // for inference to see how passing embeddings in a batch works.
    llama_batch embd_batch = llama_batch_init(1, 0, 1);
    embd_batch.embd = embd;
    embd_batch.pos[0] = 0;
    embd_batch.n_seq_id[0] = 1;
    embd_batch.seq_id[0][0] = 0;
    embd_batch.logits[0] = true;

    if (llama_decode(inf_ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(3));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    // Sample a token (sp=sampled token)
    llama_token sp_token = llama_sampler_sample(sampler, inf_ctx, input_tokens.size()-1);
    std::string sp_str = token_as_string(model, sp_token);
    printf("%snew_token_seq1: %d : token_str1 [%s]%s\n", ORANGE, sp_token, sp_str.c_str(), RESET);

    int decode_calls = 15;

    int pos = input_tokens.size();
    std::vector<std::string> output;

    printf("%sInference:%s\n", ORANGE, RESET);
    while (decode_calls--) {
        llama_batch update_batch = llama_batch_init(1, 0, 1);
        update_batch.token[0] = sp_token;
        update_batch.pos[0] = pos++;
        update_batch.n_tokens = 1;

        update_batch.n_seq_id[0] = 1;
        update_batch.seq_id[0][0] = 0;
        update_batch.logits[0] = true;

        if (llama_decode(inf_ctx, update_batch) != 0) {
            fprintf(stderr, "llama_decode() failed\n");
            return 1;
        }

        sp_token = llama_sampler_sample(sampler, inf_ctx, 0);
        std::string sp_str = token_as_string(model, sp_token);
        output.push_back(sp_str);
        printf("%stoken_seq: %.4d : token [%s]%s\n", ORANGE, sp_token, sp_str.c_str(), RESET);

        llama_sampler_reset(sampler);

        llama_batch_free(update_batch);
    }
    printf("Generated output:\n");
    for (size_t i = 0; i < output.size(); i++) {
        printf("%s%s%s", GREEN, output[i].c_str(), RESET);
    }
    printf("\n");

    llama_batch_free(batch);
    llama_free(inf_ctx);
    llama_free(embd_ctx);
    llama_free_model(model);
    llama_backend_free();
    llama_sampler_free(sampler);

    return 0;
}
