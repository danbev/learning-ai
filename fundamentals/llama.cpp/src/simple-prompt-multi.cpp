#include "llama.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>

std::string get_token_as_string(llama_token token, const llama_vocab* vocab) {
    if (token == llama_vocab_eos(vocab)) {
        fprintf(stderr, "\n");
        fflush(stderr);
        return "<eos>";
    }
    int lsplit = 0;
    bool special = false;
    std::vector<char> piece(8, 0);
    int n_tokens = llama_token_to_piece(vocab, token, piece.data(), piece.size(), lsplit, special);
    if (n_tokens < 0) {
        piece.resize(-n_tokens);
        llama_token_to_piece(vocab, token, piece.data(), piece.size(), lsplit, special);
    } else {
        piece.resize(n_tokens);
    }

    return std::string(piece.data(), piece.size());
}

int main(int argc, char** argv) {
    llama_model_params model_params = llama_model_default_params();
    int main_gpu = 0;
    int num_gpu_layers = 0;
    std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";

    model_params.main_gpu = main_gpu;
    model_params.n_gpu_layers = num_gpu_layers;
    fprintf(stdout, "llama.cpp example using model: %s\n", model_path.c_str());

    std::string prompt = "Hello ";
    std::string prompt2 = "Dan loves ice cream";

    ggml_backend_load_all();
    llama_backend_init();

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    ctx_params.n_seq_max = 2;

    struct llama_sampler* sampler = llama_sampler_init_greedy();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // Tokenize the prompt.
    const int add_bos_token = llama_vocab_get_add_bos(vocab);
    const bool add_bos  = add_bos_token != -1 ? bool(add_bos_token) :
        (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_SPM); // SPM = SentencePiece Model

    printf("add_bos: %d\n", add_bos);
    printf("prompt.len: %ld\n", prompt.length());
    int input1_len = prompt.length();
    std::vector<llama_token> input_tokens(input1_len);
    int n_tokens = llama_tokenize(vocab,
                              prompt.data(),
                              prompt.length(),
                              input_tokens.data(),
                              input_tokens.size(),
                              true,
                              false);
    // llama_tokenize will return the negative length of the token if
    // it is longer that the passed in result.length. If that is the case
    // then we need to resize the result vector to the length of the token
    // and call llama_tokenize again.
    if (n_tokens < 0) {
        input_tokens.resize(-n_tokens);
        llama_tokenize(vocab, prompt.data(), prompt.length(), input_tokens.data(), input_tokens.size(), add_bos, false);
    } else {
        input_tokens.resize(n_tokens);
    }
    fprintf(stderr, "\n");
    fprintf(stdout, "seq_0 n_tokens: %d\n", n_tokens);

    int input2_len = prompt2.length() + add_bos;
    std::vector<llama_token> input_tokens2(input2_len);
    int n_tokens2 = llama_tokenize(vocab,
                              prompt2.data(),
                              prompt2.length(),
                              input_tokens2.data(),
                              input_tokens2.size(),
                              true,
                              false);
    if (n_tokens2 < 0) {
        input_tokens2.resize(-n_tokens2);
        llama_tokenize(vocab, prompt2.data(), prompt2.length(), input_tokens2.data(), input_tokens2.size(), add_bos, false);
    } else {
        input_tokens2.resize(n_tokens2);
    }
    fprintf(stdout, "seq_1 n_tokens: %d\n", n_tokens2);

    // Create a new batch
    llama_batch batch = llama_batch_init(512, /*embd*/ 0, /*n_seq_max*/ 2);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = input_tokens[i];
        batch.pos[i] = i,
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
        batch.n_tokens++;
    }
    batch.logits[batch.n_tokens - 1] = true;
    int seq_0_cur = n_tokens;

    int pos = batch.n_tokens;
    for (int i = 0; i < n_tokens2; i++) {
        int idx = pos + i;
        batch.token[idx] = input_tokens2[i];
        batch.pos[idx] = i,
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 1;
        batch.logits[idx] = false;
        batch.n_tokens++;
    }
    batch.logits[batch.n_tokens - 1] = true;
    int seq_1_cur = n_tokens2;

    fprintf(stderr, "batch.n_tokens: %d\n", batch.n_tokens);
    fprintf(stderr, "batch.tokens: [");
    for (int i = 0; i < batch.n_tokens; i++) {
        fprintf(stderr, "%d, ", batch.token[i]);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "prompt seq0: %s\n", prompt.c_str());
    fprintf(stderr, "prompt seq1: %s\n", prompt2.c_str());
    fflush(stderr);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    const int n_len = 30;

    int n_cur = batch.n_tokens;
    int n_decode = batch.n_tokens;

    std::vector<std::string> seq_0_output;
    std::vector<std::string> seq_1_output;
    
    while (n_cur <= n_len) {
        const llama_token new_token_id_0 = llama_sampler_sample(sampler, ctx, -2);
        const llama_token new_token_id_1 = llama_sampler_sample(sampler, ctx, -1);
        llama_batch new_batch = llama_batch_init(2, 0, 2);
        new_batch.n_tokens = 2;
        new_batch.token[0] = new_token_id_0;
        new_batch.token[1] = new_token_id_1;
        new_batch.pos[0] = seq_0_cur++;
        new_batch.pos[1] = seq_1_cur++;
        new_batch.n_seq_id[0] = 1;
        new_batch.n_seq_id[1] = 1;
        new_batch.seq_id[0][0] = 0;
        new_batch.seq_id[1][0] = 1;
        new_batch.logits[0] = true;
        new_batch.logits[1] = true;

        // Sequence 1
        {
            std::string token_str = get_token_as_string(new_token_id_0, vocab);
            if (n_cur == n_len || token_str == "<eos>") {
                fprintf(stderr, "\n");
                fflush(stderr);
                break;
            }
            seq_0_output.push_back(std::move(token_str));
        }

        // Sequence 2
        {
            std::string token_str = get_token_as_string(new_token_id_1, vocab);
            if (n_cur == n_len || token_str == "<eos>") {
                fprintf(stderr, "\n");
                fflush(stderr);
                break;
            }
            seq_1_output.push_back(std::move(token_str));
        }
        n_decode += 1;
        n_cur += 1;

        if (llama_decode(ctx, new_batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        llama_batch_free(new_batch);
    }

    fprintf(stdout, "\nSequence 0:");
    for (auto str : seq_0_output) {
        fprintf(stdout, "%s", str.c_str());
    }

    fprintf(stdout, "\n\nSequence 1:");
    for (auto str : seq_1_output) {
        fprintf(stdout, "%s", str.c_str());
    }

    fprintf(stdout, "\n\nDecoded %d tokens\n", n_decode);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
