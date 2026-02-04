#include <stdio.h>

#include "ggml.h"
#include "ggml-cpu.h"

int main(int argc, char** argv) {
    printf("GGML build_forward_select example\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * two = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_f32(one, 2.0f);
    ggml_set_f32(two, 3.0f);

    struct ggml_tensor * op_add = ggml_add(ctx, one, two);
    ggml_set_name(op_add, "op_add");

    struct ggml_tensor * op_mul = ggml_mul(ctx, one, two);
    ggml_set_name(op_mul, "op_mul");

    struct ggml_cgraph * c_graph = ggml_new_graph(ctx);

    // Create an array with all the tensors that we want to have included in
    // in the graph, though only one will be active.
    struct ggml_tensor * tensors[] = {op_add, op_mul};
    int n_tensors = 2;
    int active_operation_idx = 1; // change to 0 for addition operation and see different result

    // Notice that in contrast to ggml_build_forward_expand this function returns
    // the selected "active" tensor, whereas ggml_build_forward_expand does not return void.
    struct ggml_tensor * result = ggml_build_forward_select(c_graph, tensors, n_tensors, active_operation_idx);

    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, c_graph, 1);
    if (st != GGML_STATUS_SUCCESS) {
        printf("could not compute graph\n");
        return 1;
    }

    float val = ggml_get_f32_1d(result, 0);
    printf("Inputs: %.1f, %.1f\n", ggml_get_f32_1d(one, 0), ggml_get_f32_1d(two, 0));
    printf("Selected Operation Index: %d (0=Add, 1=Mul)\n", active_operation_idx);
    printf("Computed Result: %.1f\n", val);

    ggml_free(ctx);
    return 0;
}
