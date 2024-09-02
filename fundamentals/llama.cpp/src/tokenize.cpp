#include "llama.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>


std::string token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(llama_get_model(ctx), token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

int main(int argc, char** argv) {
    llama_model_params model_params = llama_model_default_params();

    model_params.main_gpu = 0;
    std::string model_path;

    if (argc > 1) {
        model_path = argv[1];
    } else {
        fprintf(stdout, "No model path provided. Please specify a model to be used\n");
    }

    fprintf(stdout, "Using model: %s\n", model_path.c_str());

    std::string prompt = "ÅWhat is LoRA?";

    llama_backend_init();

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // Tokenize the prompt.
    const int add_bos_token = llama_add_bos_token(model);
    const bool add_bos  = add_bos_token != -1 ? bool(add_bos_token) :
        (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM); // SPM = SentencePiece Model

    printf("add_bos: %d\n", add_bos);
    printf("prompt.len: %ld\n", prompt.length());
    int n_tokens = prompt.length() + add_bos;
    std::vector<llama_token> input_tokens(n_tokens);
    n_tokens = llama_tokenize(model,
                              prompt.data(),
                              prompt.length(),
                              input_tokens.data(),
                              input_tokens.size(),
                              true,
                              true);
    // llama_tokenize will return the negative length of the token if
    // it is longer that the passed in result.length. If that is the case
    // then we need to resize the result vector to the length of the token
    // and call llama_tokenize again.
    if (n_tokens < 0) {
        input_tokens.resize(-n_tokens);
        int new_len = llama_tokenize(model,
                prompt.data(),
                prompt.length(),
                input_tokens.data(),
                input_tokens.size(),
                add_bos,
                false);
    } else {
        input_tokens.resize(n_tokens);
    }
    fprintf(stderr, "\n");
    fprintf(stdout, "n_tokens: %d\n", n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        fprintf(stdout, "%s ", token_to_piece(ctx, input_tokens[i], true).c_str());
    }
    printf("\n");

    char detokenized[1024];
    int32_t ret = llama_detokenize(model, input_tokens.data(), n_tokens, detokenized, 1024, false, false);
    printf("Detokenized: %s\n", detokenized);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
