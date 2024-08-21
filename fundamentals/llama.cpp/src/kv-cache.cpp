#include "llama.h"

#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    fprintf(stdout, "llama.cpp KV-Cache exploration\n");
    llama_model_params model_params = llama_model_default_params();

    model_params.main_gpu = 0;
    model_params.n_gpu_layers = 0;
    std::string model_path = "models/llama-2-13b-chat.Q4_0.gguf";
    fprintf(stdout, "llama.cpp example using model: %s\n", model_path.c_str());

    llama_backend_init();

    llama_model* model = llama_load_model_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to to load model %s\n" , __func__, model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed  = 1234;
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }
    printf("llama_context created\n");
    //struct llama_kv_cache kv_cache = ctx->kv_self;

    struct llama_kv_cache_view kv_view =  llama_kv_cache_view_init(ctx, 1);
    printf("kv_view n_cells: %d\n", kv_view.n_cells);
    printf("kv_view n_max_seq: %d\n", kv_view.n_seq_max);
    printf("kv_view token_count: %d\n", kv_view.token_count);
    printf("kv_view used_cells: %d\n", kv_view.used_cells);

    struct llama_kv_cache_view_cell* cells = kv_view.cells;
    for (int i = 0; i < kv_view.n_cells; i++) {
        printf("cell[%d] pos: %d\n", i, cells[i].pos);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
