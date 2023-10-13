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

  return 0;
}
