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

  std::vector<llama_token> tokens_list;
  std::string query = "What is LoRA?";
  std::cout << "query: " << query << std::endl;

  tokens_list = ::llama_tokenize(ctx, query, true);
  std::cout << "tokens: " << std::endl;
  for (auto token : tokens_list) {
    std::cout << token << " :" << llama_token_to_piece(ctx, token) << std::endl;
  }

  return 0;
}
