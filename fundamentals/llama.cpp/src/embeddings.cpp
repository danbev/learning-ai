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

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;

    // Chat/Instruct model usage.
    //std::string model_path = "models/llama-2-7b-chat.Q4_K_M.gguf";
    //std::string prompt = "<s>[INST] <<SYS>>\n\n<</SYS>>\n\nWhat is LoRA? [/INST]";

    // Base model usage
    std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";
    //std::string model_path = "models/mamba-1.4b-f16.gguf";
    std::string prompt = "What is LoRA?";

    printf("%sprompt: %s%s\n", GREEN, prompt.c_str(), RESET);

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    ctx_params.embeddings = true;

    llama_context* embd_ctx = llama_new_context_with_model(model, ctx_params);
    if (embd_ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    std::vector<llama_token> input_tokens = tokenize_prompt(model, prompt);
    llama_batch prompt_batch = create_batch(input_tokens.size(), input_tokens, model);

    // Decode the prompt to generate the embeddings. We are not going to use
    // the logits at this stage.
    if (llama_decode(embd_ctx, prompt_batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    // Now we will extract the embeddings.
    int n_embd = llama_n_embd(model);
    std::vector<float> token_embeddings;

    float* embd = llama_get_embeddings(embd_ctx);
    token_embeddings.insert(token_embeddings.end(), embd, embd + input_tokens.size() * n_embd);

    /* Alternative way to extract embeddings.
    for (size_t i = 0; i < input_tokens.size(); i++) {
        float* embd = llama_get_embeddings_ith(embd_ctx, i);
        token_embeddings.insert(token_embeddings.end(), embd, embd + n_embd);

        printf("Original embedding %zu: ", i);
            for (int j = 0; j < 5; j++) {
            printf("%f ", embd[j]);
        }
        printf("\n");
    }
    */

    // Print out the first 5 embeddings from all token embeddings generated.
    for (size_t i = 0; i < input_tokens.size(); i++) {
        printf("%sembedding %ld \%s", BLUE, i, RESET);
        float* token_embd = token_embeddings.data() + (i * n_embd);
        for (int j = 0; j < 5; j++) {
            printf("%s%10f%s ", BLUE, token_embd[j], RESET);
        }
        printf("\n");
    }

    // Now we are done with the context used to generate the embeddings. This
    // is to simulate a case where the embeddings were generated as a previous
    // stage for usage later.
    llama_kv_cache_clear(embd_ctx);
    llama_free(embd_ctx);
    llama_batch_free(prompt_batch);

    // Now we are going to create a new context for inference.
    llama_context_params inf_ctx_params = llama_context_default_params();
    inf_ctx_params.n_threads = 4;
    inf_ctx_params.n_threads_batch = 4;
    llama_context* inf_ctx = llama_new_context_with_model(model, inf_ctx_params);
    int pos = 0;

    // Next we create a batch for the token embeddings generated above.
    // The following is creating a single batch with 6 token embeddings in it.
    llama_batch embd_batch = llama_batch_init(input_tokens.size(), n_embd, 1);
    embd_batch.n_tokens = input_tokens.size();
    embd_batch.embd = token_embeddings.data();
    printf("%sToken embeddings size: %d, n_tokens: %d%s\n", GREEN, n_embd, embd_batch.n_tokens, RESET);
    for (size_t i = 0; i < input_tokens.size(); i++) {
        embd_batch.pos[i] = i; 
        embd_batch.n_seq_id[i] = 1;
        embd_batch.seq_id[i][0] = 0;
        embd_batch.logits[i] = i == input_tokens.size() - 1;
    }
    printf("%slast position : %d%s\n", GREEN, embd_batch.pos[input_tokens.size() - 1], RESET);

    // Decode the token embeddings to generate the logits.
    if (llama_decode(inf_ctx, embd_batch) != 0) {
        fprintf(stderr, "llama_decode() failed for token\n");
        return 1;
        llama_batch_free(embd_batch);
    }
    pos = embd_batch.pos[input_tokens.size() - 1];

    float* logits = llama_get_logits(inf_ctx);
    printf("Top 5 logits:\n");
    std::vector<std::pair<llama_token, float>> top_logits;
    for (int i = 0; i < llama_n_vocab(model); i++) {
	top_logits.push_back(std::make_pair(i, logits[i]));
    }
    std::partial_sort(top_logits.begin(),
	      top_logits.begin() + 5,
	      top_logits.end(),
	      [](const std::pair<llama_token, float>& a,
		 const std::pair<llama_token, float>& b) {
		  return a.second > b.second;
	      });
    for (int i = 0; i < 5; i++) {
    printf("Token %d (%s): %f\n",
       top_logits[i].first,
       token_as_string(model, top_logits[i].first).c_str(),
       top_logits[i].second);
    }

    // Next create a sampler chain for sampling the next token.
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(1.5));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(10));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    //llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));
    //llama_sampler_chain_add(sampler, llama_sampler_init_softmax());
    //llama_sampler_chain_add(sampler, llama_sampler_init_top_k(3));

    std::vector<std::string> output;
    // Sample a token (sp=sampled token)
    llama_token sp_token = llama_sampler_sample(sampler, inf_ctx, -1);
    std::string sp_str = token_as_string(model, sp_token);
    printf("%stoken_seq: %d : token_str [%s]%s\n", ORANGE, sp_token, sp_str.c_str(), RESET);
    output.push_back(sp_str);

    int decode_calls = 5;
    while (decode_calls--) {
        llama_batch update_batch = llama_batch_init(1, 0, 1);
        update_batch.n_tokens = 1;
        update_batch.token[0] = sp_token;
        update_batch.pos[0] = ++pos;

        update_batch.n_seq_id[0] = 1;
        update_batch.seq_id[0][0] = 0;
        update_batch.logits[0] = true;
        printf("%sInference: token: %d, pos: %d %s\n", ORANGE, update_batch.token[0], update_batch.pos[0], RESET);

        if (llama_decode(inf_ctx, update_batch) != 0) {
            fprintf(stderr, "llama_decode() failed\n");
            return 1;
        }

        float* logits = llama_get_logits(inf_ctx);
        printf("Top 5 logits:\n");
        std::vector<std::pair<llama_token, float>> top_logits;
        for (int i = 0; i < llama_n_vocab(model); i++) {
            top_logits.push_back(std::make_pair(i, logits[i]));
        }
        std::partial_sort(top_logits.begin(),
                  top_logits.begin() + 5,
                  top_logits.end(),
                  [](const std::pair<llama_token, float>& a,
                     const std::pair<llama_token, float>& b) {
                      return a.second > b.second;
                  });
        for (int i = 0; i < 5; i++) {
        printf("Token %d (%s): %f\n",
           top_logits[i].first,
           token_as_string(model, top_logits[i].first).c_str(),
           top_logits[i].second);
        }

        sp_token = llama_sampler_sample(sampler, inf_ctx, 0);
        sp_str = token_as_string(model, sp_token);
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

    llama_free(inf_ctx);
    llama_free_model(model);
    llama_backend_free();
    llama_sampler_free(sampler);

    return 0;
}
