#include "llama.h"
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

struct token_position {
    size_t seq_id;
    size_t index;
    token_position() : seq_id(0), index(0) {}
    token_position(size_t s, size_t i) : seq_id(s), index(i) {}

    std::string to_string() const {
        return "{ seq_id: " + std::to_string(seq_id) + ", index: " + std::to_string(index) + " }";
    }
};

std::unordered_map<llama_token, std::vector<token_position>> find_common_tokens(
        const std::vector<std::vector<llama_token>>& input_tokens,
        llama_model* model) {
    if (input_tokens.empty()) {
        return {};
    }

    std::unordered_map<llama_token, std::unordered_map<size_t, token_position>> token_positions;
    for (size_t seq_id = 0; seq_id < input_tokens.size(); ++seq_id) {
        const auto& current_vec = input_tokens[seq_id];
        for (size_t token_idx = 0; token_idx < current_vec.size(); ++token_idx) {
            llama_token token = current_vec[token_idx];
            if (token_positions[token].find(seq_id) == token_positions[token].end()) {
                token_positions[token][seq_id] = token_position(seq_id, token_idx);
            }
        }
    }

    std::unordered_map<llama_token, std::vector<token_position>> common_tokens;
    for (const auto& entry : token_positions) {
        if (llama_add_bos_token(model) && entry.first == 1) {
            continue;
        }
        if (entry.second.size() > 1) {
            std::vector<token_position> positions;
            positions.reserve(entry.second.size());
            for (const auto& seq_pos : entry.second) {
                positions.push_back(seq_pos.second);
            }
            common_tokens[entry.first] = std::move(positions);
        }
    }

    return common_tokens;
}

void print_common_tokens(std::unordered_map<llama_token, std::vector<token_position>> common_tokens) {
    for (const auto& token_info : common_tokens) {
        printf("Token id [%d] in common at positions:\n", token_info.first);
        for (const auto& pos : token_info.second) {
            printf("  Sequence %zu, Index %zu\n", pos.seq_id, pos.index);
        }
    }
}

llama_batch create_batch(int size, std::vector<std::vector<llama_token>> input_tokens, llama_model* model) {
    int n_prompts = input_tokens.size();
    printf("Creating new llama_batch with %d sequences\n", n_prompts);

    auto common_tokens = find_common_tokens(input_tokens, model);
    if (common_tokens.empty()) {
        printf("No common tokens found. Beginning of Sequence (bos) is not considered\n");
    } else {
        print_common_tokens(common_tokens);
    }

    // Create a single batch for both prompts.
    llama_batch batch = llama_batch_init(size, 0, n_prompts);

    for (size_t s = 0; s < input_tokens.size(); s++) {
        printf("Processing prompt %ld, batch_n_tokens: %d \n", s, batch.n_tokens);
        std::vector<llama_token> prompt_tokens = input_tokens[s];
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
	    int token_id = prompt_tokens[i];
            int idx = batch.n_tokens;
            batch.token[idx] = token_id;
            batch.pos[idx] = i;

	    auto it = common_tokens.find(token_id);
	    if (it != common_tokens.end()) {
                std::vector<token_position> tps = it->second;
		batch.n_seq_id[idx] = tps.size();
		batch.seq_id[idx][0] = s;
		for (size_t j = 1; j < tps.size(); j++) {
		    auto tp = tps[j];
		    if (tp.seq_id != s) {
			printf("seq: %zu, token %d: %s \n", s, token_id, tps[j].to_string().c_str());
			batch.seq_id[idx][j] = tps[j].seq_id;
		    }
		}
	    } else {
		batch.n_seq_id[idx] = 1;
		batch.seq_id[idx][0] = s;  // the sequence id
	    }
            batch.logits[idx] = i == prompt_tokens.size() - 1;

            batch.n_tokens++;
            printf("idx: %4d, token: %6d, seq_id: %ld, logits: %d\n", idx, token_id, s, batch.logits[idx]);
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
    //std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";
    std::string model_path = "models/mamba-1.4b-f16.gguf";

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;

    // This prompt is 69 tokens
    //std::string prompt1 = R"(You are an AI assistant specializing in task completion. Your goal is to provide clear, concise, and accurate responses to user queries. Always maintain a helpful and professional tone. If a request is unclear, ask for clarification. Prioritize user safety and ethical considerations in your answers.)";
    std::string prompt1 = "Yesterday was Friday";
    //std::string prompt2 = "How many r's are there in strawberry?";
    std::string prompt2 = "Tomorrow was Monday";

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
    ctx_params.n_seq_max = 2;
    ctx_params.n_batch = 80;
    ctx_params.n_ubatch = 32;

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

    llama_batch batch = create_batch(512, {input_tokens1, input_tokens2}, model);
    print_batch(batch);

    /*
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    int embd_size = llama_n_embd(model);
    printf("logits for prompt1:\n");
    float* logits1 = llama_get_logits_ith(ctx, input_tokens1.size() - 1);
    for (int i = embd_size - 10; i < embd_size; i++) {
        fprintf(stderr, "logits1[%d]: %f\n", i, logits1[i]);
    }
    printf("logits for prompt2:\n");
    float* logits2 = llama_get_logits_ith(ctx, input_tokens1.size() + input_tokens2.size() - 1);
    for (int i = embd_size - 10; i < embd_size; i++) {
        fprintf(stderr, "logits2[%d]: %f\n", i, logits2[i]);
    }
    */

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
