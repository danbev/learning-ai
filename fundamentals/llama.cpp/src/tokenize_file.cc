#include "train.h"
#include "llama.h"

/*
   This is a standalone example of using llama.cpp training tokenize_file
   function. Is is only intended to be used to verify that that format of
   training data is correct.

   To inspect the samples, comment in the following line in train.cpp:
   printf("sample: '%s'\n", buf_sample.data());
*/
int main() {
    std::string training_data = "data/assistent-training.txt";
    std::string model = "models/llama-2-7b-chat.gguf";

    std::vector<llama_token> train_tokens;
    std::vector<size_t> train_samples_begin;
    std::vector<size_t> train_samples_size;

    struct llama_model_params llama_mparams = llama_model_default_params();
    llama_mparams.n_gpu_layers = 27;
    llama_mparams.vocab_only = false;

    struct llama_model* lmodel = llama_load_model_from_file(model.c_str(), llama_mparams);

    struct llama_context_params llama_cparams = llama_context_default_params();
    struct llama_context* lctx = llama_new_context_with_model(lmodel, llama_cparams);

    size_t ret = tokenize_file(lctx,
        training_data.c_str(),
        "<s>",  // sample start token
        true, // include sample start
        false, // overlapping
        70, // n_tokens
        train_tokens,
        train_samples_begin,
        train_samples_size);

    printf("%s: tokenize_file returned %zu\n", __func__, ret);
    return 0;
}
