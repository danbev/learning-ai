#include "llama.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    llama_model_params model_params = llama_model_default_params();

    // parse the two optional integers named "main_gpu" and "n_gpu_layers" and set the default to zero if they are not provided.
    int main_gpu = 0;
    int num_gpu_layers = 0;
    std::string model_path = "models/llama-2-7b.Q4_K_M.gguf";

    if (argc > 1) {
        main_gpu = atoi(argv[1]);
    }
    if (argc > 2) {
        num_gpu_layers = atoi(argv[2]);
    }
    if (argc > 3) {
        model_path = argv[3];
    }

    model_params.main_gpu = main_gpu;
    model_params.n_gpu_layers = num_gpu_layers;
    fprintf(stdout, "llama.cpp example using model: %s\n", model_path.c_str());

    // If the prompt provided is in the form of a question like it is here
    // the model will predict the first token to be a new line, completing the
    // prompt with a new line. It will then predict the next token to be the
    // another new line.
    std::string prompt = "What is LoRA?";
    std::string prompt2 = "Dan loves ice cream";

    llama_backend_init();
    //llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

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
    fprintf(stdout, "n_tokens: %d\n", n_tokens);

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
    printf("prompt2.len: %ld\n", prompt2.length());

    // Create a new batch
    llama_batch batch = llama_batch_init(512,/*embd*/ 0, /*n_seq_max*/ 2);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = input_tokens[i];
        batch.pos[i] = i,
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;  // the sequence id
        batch.logits[i] = false;
        batch.n_tokens++;
    }

    int pos = batch.n_tokens;
    for (int i = 0; i < n_tokens2; i++) {
	int idx = pos + i;
        batch.token[idx] = input_tokens2[i];
        batch.pos[idx] = idx,
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 1;
        batch.logits[idx] = false;
        batch.n_tokens++;
    }
    // Instruct llama to generate the logits for the last token
    batch.logits[batch.n_tokens - 1] = true;
    fprintf(stderr, "batch.n_tokens: %d\n", batch.n_tokens);
    fprintf(stderr, "batch.tokens: [");
    for (int i = 0; i < batch.n_tokens; i++) {
        fprintf(stderr, "%d, ", batch.token[i]);
    }
    fprintf(stderr, "]\n");

    fprintf(stderr, "prompt: %s", prompt.c_str());
    fflush(stderr);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode() failed\n");
        return 1;
    }

    const int n_len = 80;

    int n_cur = batch.n_tokens;
    int n_decode = batch.n_tokens;
    
    while (n_cur <= n_len) {
        const llama_token new_token_id = llama_sampler_sample(sampler, ctx, -1);
        // This is the token id that the model predicted.

        // is it an end of stream?
        if (new_token_id == llama_vocab_eos(vocab) || n_cur == n_len) {
            fprintf(stderr, "\n");
            fflush(stderr);
            break;
        }

        int lsplit = 0;
        bool special = false;
        std::vector<char> piece(8, 0);
        int n_tokens = llama_token_to_piece(vocab, new_token_id, piece.data(), piece.size(), lsplit, special);
        if (n_tokens < 0) {
            piece.resize(-n_tokens);
            llama_token_to_piece(vocab, new_token_id, piece.data(), piece.size(), lsplit, special);
        } else {
            piece.resize(n_tokens);
        }
        std::string piece_str = std::string(piece.data(), piece.size());
        fprintf(stderr, "%s", piece_str.c_str());
        fflush(stderr);

        llama_batch single_token_batch = llama_batch_init(1,/*embd*/ 0, /*n_seq_max*/ 1);
        single_token_batch.n_tokens = 1; // We are only passing in one token.
        single_token_batch.token[0] = new_token_id; // the new token id.
        single_token_batch.pos[0] = n_cur, // the position in the sequence.
        single_token_batch.n_seq_id[0] = 1;  // the number of sequences for this token.
        single_token_batch.seq_id[0][0] = 1; // the actual sequence id.
        single_token_batch.logits[0] = true;

        n_decode += 1;
        n_cur += 1;

        // With the new token added to the batch, we can now predict the next token.
        if (llama_decode(ctx, single_token_batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
        llama_batch_free(single_token_batch);
    }
    fprintf(stdout, "\nDecoded %d tokens\n", n_decode);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
