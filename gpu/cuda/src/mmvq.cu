// Simplified version of llama.cpp's mul_mat_vec_q kernel (ggml/src/ggml-cuda/mmvq.cu)
//
// It simulates the two cases from llama.cpp issue #25618 on GB10:
//   - single-token decode: ncols = 1, 8 warps (the halve_iters config)
//   - 8-token speculative verification batch: ncols = 8, 2 warps
// and checks whether token 0 gets bit-identical results in both.
//
// The weight format is simplified: blocks of 256 int8 values with one float
// scale, instead of Q4_K. With qi = 64 and vdr = 4 this gives qi/vdr = 16
// threads per weight block, the same as Q4_K (qi = 32, vdr = 2), so the loop
// trip counts match llama.cpp for K = 4096.
//
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

constexpr int WARP_SIZE = 32;

constexpr int QK_W = 256;
struct block_w {
    float  d;
    int8_t qs[QK_W];
};

constexpr int qk = QK_W;     // values per weight block
constexpr int qi = QK_W / 4; // 32-bit ints per weight block (4 int8 per int) = 64

constexpr int QK_X = 32;
struct block_x {
    float  d;
    int8_t qs[QK_X];
};

// Vector dot ratio (vdr), how many ints one thread handles per vec_dot call.
constexpr int vdr = 4;

// Butterfly reduction: after this every lane holds the sum of all 32 lanes.
__device__ float warp_reduce_sum(float x) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset);
    }
    return x;
}

__device__ float vec_dot(const block_w * bw, const block_x * bx, int kqs) {
    // get a pointer to the quantized weight values, the array qs.
    const int * w = (const int *) bw->qs;

    // The activation bx is 32 values, and each thread uses kqs to find its
    // activation block.
    const block_x * xb = &bx[kqs / (QK_X/4)];
    // Get a pointer to the quantized values of the activation block.
    const int * x = (const int *) xb->qs;
    // Get an offset into that activation block.
    const int x_off = kqs % (QK_X/4);         // int offset inside that block: 0 or 4

    // Integer math is exact, so the order here does not matter.
    int sumi = 0;
    for (int i = 0; i < vdr; ++i) {
        sumi = __dp4a(w[kqs + i], x[x_off + i], sumi);
    }
    // The scales are applied once, at the end, in float.
    return bw->d * xb->d * (float) sumi;
}

template <int ncols, int nwarps>
__global__ void mul_mat_vec(const block_w * W, const block_x * X, float * dst, int ncols_x, int nrows) {
    // vdr=4, WARP_SIZE=32, qi=64
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
    // nwarps 8: 4 * 8 * 32 / 64 = 16 
    // nwarps 2: 4 * 2 * 32 / 64 = 4

    // Each thread will process a row in the weights matrix (4096x4096)
    // where each row contains 4096 values split into 16, 256 value blocks.
    // block (0,0,0) -> row 0
    // block (1,0,0) -> row 1
    // ...
    const int row = blockIdx.x;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;

    const int blocks_per_row = ncols_x / qk;    // 4096/256 = 16 weight blocks
    // Pointer to this threads row.
    const block_w * w_row = W + row * blocks_per_row;

    const int blocks_per_col = ncols_x / QK_X;  // 4096/32 = 128 activation blocks per token

    float tmp[ncols] = {0.0f};
    // The weight matrix will be processed like this:
    // [0         256       512                            3840  4096]
    // { kbx=0     }{   kbx=1 }{ kbx=2}   ...                 { kbx=15}
    // th: 0-15   th: 16-31     th: 32-47                   th: 240-255
    //
    // Notice that 16 threads share a weight block which consists of 256 value.
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
	// So we are processing a single weight block of 256 values here.
	// qs as ints: [0 1 2 3][4 5 6 7][8 9 10 11]...[60 61 62 63]
	// kqs:            0        1        2             60
	// thread:      (0,0,0,  (1,0,0)  (2,0,0)       (15,0,0)
	// values:      0...15   16...31  32...47       240...255

	// kqs is the index of the first int this thread reads inside the weight
	// blocks qs (quants) array. And each thread reads 4 (vdr) ints from there.
	// kqs = vdr * (tid % 16. For example thread 1 will start reading from
	// index 4, values 16..31.
	// So thread (1,0,0) computes W[0][16...31] * x[16...31]
        const int kqs = vdr * (tid % (qi/vdr));

	// Each activation block has 32 values (QK_X=32, qs[32]), so 8 ints and
	// covers two threads. One thread will process 4 weight ints, for example
	// thread 0 will process 0..15. 
	// x (token 0): [0        32      63 ...                       4096]
	//              {t0  }{t1 }{t2 }{t3 }                          {   }
	//              {kby=0    }{kby=1   }                          kby=128 
        const int kby = kbx * (qk/QK_X);

	// When we have 8 warps the blocks per iteration will be 16 and the
	// outer loop will only run once. So we will only really be assigning to
	// tmp[j], there will be no addition operation. But if we have 2 warps
	// only then there will be 4 iterations through the outer loop and we
	// will have 4 additions to tmp[j].
        //
	// For example:
        // 4 threads, 1 loop pass. Each thread holds one term and nothing is
	// added in the loop. The the combining step adds them up:
        // thread 0: t0 = 1.00
        // thread 1: t1 = 0.004
        // thread 2: t2 = 0.004
        // thread 3: t3 = 0.004
        // combine: 1.00 + 0.004 = 1.004 -> rounds to 1.00
        // 1.00 + 0.004 = 1.004 -> rounds to 1.00
        // 1.00 + 0.004 = 1.004 -> rounds to 1.00
        // result: 1.00
        //
        // Now with 2 threads, we have two loop passes:
        // thread 0 loop: tmp = t0 + t2 = 1.00  + 0.004 = 1.004 -> 1.00 (rounded)
        // thread 1 loop: tmp = t1 + t3 = 0.004 + 0.004 = 0.008
        // combine: 1.00 + 0.008 = 1.008 -> 1.01
        // result: 1.01
        for (int j = 0; j < ncols; ++j) {
            tmp[j] += vec_dot(&w_row[kbx], &X[j * blocks_per_col + kby], kqs);
        }
    }

    // In each warp except 0 copies the results to shared memory (one copy per block).
    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols][WARP_SIZE];
    if (threadIdx.y > 0) {
        for (int j = 0; j < ncols; ++j) {
            tmp_shared[threadIdx.y-1][j][threadIdx.x] = tmp[j];
        }
    }

    __syncthreads();

    if (threadIdx.y > 0) {
        return;
    }

    for (int j = 0; j < ncols; ++j) {
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[j] += tmp_shared[l][j][threadIdx.x];
        }
        // Sum the 32 lanes of warp 0.
        tmp[j] = warp_reduce_sum(tmp[j]);
    }

    if (threadIdx.x == 0) {
        for (int j = 0; j < ncols; ++j) {
            dst[j*nrows + row] = tmp[j];
        }
    }
}

template <int ncols, int nwarps>
static std::vector<float> run(const block_w * d_w, const block_x * d_x, float * d_dst, int ncols_x, int nrows) {
    const int blocks_per_row  = ncols_x / qk;

    const int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
    const int iters           = (blocks_per_row + blocks_per_iter - 1) / blocks_per_iter;
    printf("ncols = %d, nwarps = %d: %3d threads/block, %2d weight blocks per pass, %d loop pass(es)\n",
            ncols, nwarps, nwarps*WARP_SIZE, blocks_per_iter, iters);

    dim3 grid(nrows);
    dim3 block(WARP_SIZE, nwarps);
    mul_mat_vec<ncols, nwarps><<<grid, block>>>(d_w, d_x, d_dst, ncols_x, nrows);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(nrows);
    CUDA_CHECK(cudaMemcpy(out.data(), d_dst, nrows*sizeof(float), cudaMemcpyDeviceToHost));

    return out;
}

static void compare(const char * label, const std::vector<float> & a, const std::vector<float> & b) {
    int   ndiff    = 0;
    float max_diff = 0.0f;
    printf("\n%s:\n", label);
    for (size_t r = 0; r < a.size(); ++r) {
        uint32_t ua, ub;
        memcpy(&ua, &a[r], sizeof(ua));
        memcpy(&ub, &b[r], sizeof(ub));
        if (ua != ub) {
            if (ndiff < 3) {
                printf("  row %4zu: %.9g (0x%08x) vs %.9g (0x%08x)\n", r, a[r], ua, b[r], ub);
            }
            ndiff++;
            max_diff = fmaxf(max_diff, fabsf(a[r] - b[r]));
        }
    }
    printf("  %d/%zu rows differ, max diff = %g\n", ndiff, a.size(), max_diff);
}

int main() {
    const int ncols_x = 4096;        // K: hidden size, like attn_q in Qwen3-8B
    const int nrows   = 4096;        // M: output rows
    constexpr int NTOKENS = 8;       // verification batch: last token + 7 drafts

    std::mt19937 rng(18);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Weights matrix 4096x4096=16777216
    //    0 [0                           4095]
    //    1 [0                           4095]
    //                    ...
    // 4095 [0                           4095]
    //
    // 4096/256 = 16 blocks per row
    // Total: 16777216
    std::vector<block_w> h_w(nrows * ncols_x / QK_W);
    for (block_w & b : h_w) {
        float v[QK_W];
        float amax = 0.0f;
        for (int i = 0; i < QK_W; ++i) {
            v[i] = dist(rng);
            amax = fmaxf(amax, fabsf(v[i]));
        }
        b.d = amax / 127.0f;
        for (int i = 0; i < QK_W; ++i) {
            b.qs[i] = (int8_t) lroundf(v[i] / b.d);
        }
    }

    // Quantize random activations for NTOKENS tokens, stored token after token.
    // Token 0 is the same data in every run below.
    // Are activations will be 1024, and we have 8 tokens:
    // 0 [0                           4095]
    // 1 [0                           4095]
    //         ...
    // 7 [0                           4095]
    // Each element in h_x will store one row.
    std::vector<block_x> h_x(NTOKENS * ncols_x / QK_X);
    for (block_x & b : h_x) {
        float v[QK_X];
        float amax = 0.0f;
        for (int i = 0; i < QK_X; ++i) {
            v[i] = dist(rng);
            amax = fmaxf(amax, fabsf(v[i]));
        }
        b.d = amax / 127.0f;
        for (int i = 0; i < QK_X; ++i) {
            b.qs[i] = (int8_t) lroundf(v[i] / b.d);
        }
    }

    block_w * d_w;
    block_x * d_x;
    float   * d_dst;
    CUDA_CHECK(cudaMalloc(&d_w, h_w.size() * sizeof(block_w)));
    CUDA_CHECK(cudaMalloc(&d_x, h_x.size() * sizeof(block_x)));
    CUDA_CHECK(cudaMalloc(&d_dst, NTOKENS * nrows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(block_w), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(block_x), cudaMemcpyHostToDevice));

    // Single token decode using 1 token and 8 warps. Each warp has 32 threads
    // and we are using 8 warps so we have 256 threads in total.
    std::vector<float> single = run<1, 8>(d_w, d_x, d_dst, ncols_x, nrows);

    // 8 tokens, using 2 warps
    std::vector<float> batch = run<NTOKENS, 2>(d_w, d_x, d_dst, ncols_x, nrows);

    // Changing the number of warps changes how work is partitioned accross threads
    // and the exact order in which partial sums are added together. And recall
    // that floating point addition is not associative.
    std::vector<float> batch_8w = run<NTOKENS, 8>(d_w, d_x, d_dst, ncols_x, nrows); // 8 tokens forced to 8 warps

    compare("token 0: single (8 warps) vs batch (2 warps)", single, batch);
    compare("token 0: single (8 warps) vs batch (8 warps)", single, batch_8w);

    // Reference in double, to show that neither GPU result is more "correct" than the other.
    std::vector<double> ref(nrows, 0.0);
    const int blocks_per_row = ncols_x / QK_W;
    for (int r = 0; r < nrows; ++r) {
        for (int kb = 0; kb < blocks_per_row; ++kb) {
            const block_w & bw = h_w[r*blocks_per_row + kb];
            for (int xb = 0; xb < QK_W/QK_X; ++xb) {
                const block_x & bx = h_x[kb*(QK_W/QK_X) + xb];
                int sumi = 0;
                for (int i = 0; i < QK_X; ++i) {
                    sumi += bw.qs[xb*QK_X + i] * bx.qs[i];
                }
                ref[r] += (double) bw.d * bx.d * sumi;
            }
        }
    }
    double err_single = 0.0;
    double err_batch  = 0.0;
    for (int r = 0; r < nrows; ++r) {
        err_single = fmax(err_single, fabs(single[r] - ref[r]));
        err_batch  = fmax(err_batch,  fabs(batch[r]  - ref[r]));
    }
    printf("\nmax error vs double reference: single = %g, batch = %g\n", err_single, err_batch);

    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_dst));

    return 0;
}
