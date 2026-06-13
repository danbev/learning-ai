#include <stdio.h>
#include <string.h>
#include <cmath>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

static void print_tensor(struct ggml_tensor * t, const char * label) {
    printf("%s [%d, %d, %d, %d]\n", label, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);

    for (int b = 0; b < t->ne[3]; ++b) {
        for (int c = 0; c < t->ne[2]; ++c) {
            for (int y = 0; y < (int)t->ne[1]; y++) {
                printf("  Row %d: ", y);
                for (int x = 0; x < (int)t->ne[0]; x++) {
                    printf("%6.1f", ggml_get_f32_nd(t, x, y, c, b));
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {
    printf("GGML tensor result exploration\n");

    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };

    struct ggml_context * ctx1 = ggml_init(params);

    // first input to addition operation in context 1.
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 10);
    ggml_set_name(a, "a");
    float a_data[10] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    memcpy(a->data, a_data, ggml_nbytes(a));
    print_tensor(a, "a");

    // second input to addition operation in context 1.
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 10);
    ggml_set_name(b, "b");
    float b_data[10] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    memcpy(b->data, b_data, ggml_nbytes(b));
    print_tensor(b, "b");

    // addition operation in context 1.
    struct ggml_tensor * sum = ggml_add(ctx1, a, b);
    ggml_set_name(sum, "sum");

    struct ggml_cgraph * c_graph = ggml_new_graph(ctx1);
    ggml_build_forward_expand(c_graph, sum);

    enum ggml_status st = ggml_graph_compute_with_ctx(ctx1, c_graph, 1);
    if (st != GGML_STATUS_SUCCESS) {
        printf("Graph computation failed\n");
        return 1;
    }
    print_tensor(sum, "sum");

    // So at this point we have computed the first graph and we printed the
    // result. Now, the ideas is we now want to use this result in an second
    // graph computation. Now adding the sum ggml_tensor from above might feel
    // like something that we could do, and it actually might work depending on
    // the situation with the lifetime of the ggml_context. Just looking at it I
    // was thinking that, well I want to use the sum in my next graph computation
    // so I'll just store/pass that ggml_tensor to it and I'll be good to go:
    // struct ggml_tensor * sum2 = ggml_add(ctx2, sum, c);
    //
    // But what that does is it adds sum tensor/node operation to the new graph
    // and sum has two src tensors, a and b. And the operation it does is the
    // addition of them. So when ggml expands the nodes it will visit all the 
    // nodes and add those operations to the new graph. So it would actually
    // perform the addition operation again using a and b, it is not using the
    // result of the operation. If this was an expensive operation then this is
    // certainly not what we want (disregarding other issues where the context of
    // the first graphs might not be around and also the tensor itself could get
    // reused by).
    // Instead what we need to do is great a non-operation tensor, a leaf tensor
    // and copy the data from the sum operation into that. We can then use this
    // tensor with then second computation graph. This is what the code below
    // tries to show.

    // create the second context.
    struct ggml_context * ctx2 = ggml_init(params);

    // input tensor for the second addition operation (in context 2).
    struct ggml_tensor * c = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 10);
    ggml_set_name(c, "c");
    float c_data[10] = { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f};
    // this is just because we are using the CPU, if we had a backend we would
    // need to allocate a buffer in the backend using ggml_backend_buffer_alloc_tensor,
    // and then copy using ggml_backend_tensor_copy.
    memcpy(c->data, c_data, ggml_nbytes(c));
    print_tensor(c, "c");

    // create a leaf tensor for the result of the sum operation.
    struct ggml_tensor * sum_input = ggml_new_tensor_1d(ctx2, sum->type, sum->ne[0]);
    ggml_set_name(sum_input, "sum_input");
    // copy the data to the new tensor.
    memcpy(sum_input->data, sum->data, ggml_nbytes(sum));

    // And we can then free the first context.
    ggml_free(ctx1);

    struct ggml_tensor * sum2 = ggml_add(ctx2, sum_input, c);
    struct ggml_cgraph * c_graph2 = ggml_new_graph(ctx2);
    ggml_build_forward_expand(c_graph2, sum2);

    if (ggml_graph_compute_with_ctx(ctx2, c_graph2, 1) != GGML_STATUS_SUCCESS) {
        printf("Second Graph computation failed\n");
        return 1;
    }
    print_tensor(sum2, "sum2");

    ggml_free(ctx2);
    return 0;
}
