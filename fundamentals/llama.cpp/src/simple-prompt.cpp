#include "common.h"
#include "llama.h"

#include <iostream>

int main(int argc, char** argv) {
  std::cout << "llama.cpp example" << std::endl;
  gpt_params params;
  std::cout << "params.n_threads: " << params.n_threads << std::endl;

  return 0;
}
