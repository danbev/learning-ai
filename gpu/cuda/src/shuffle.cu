#include <cuda_runtime.h>
#include <stdio.h>

// A warp is 32 threads (lanes) that execute the same instruction together.
// Each lane has its own private registers, and __shfl_*_sync lets a lane read
// a register value from another lane in the same warp, without going through
// shared memory. Threads in a warp can't access other threads registers, it is
// the hardware that moves the data.

__global__ void shuffle_examples() {
    int lane = threadIdx.x;   // 0..31 since we launch a single warp
    int value = lane * 10;    // each lane has its own private 'value' register

    // A suffle is a pull operation. Every lane pulls the 'value' lane 3 holds.
    // (cuda-gdb) p /t 0xffffffff
    // $3 = 1111 1111 1111 1111 1111 1111 1111 1111
    int src_lane = 3;
    int bcast = __shfl_sync(0xffffffff, value, src_lane);

    // Suffle is similar to __shfl_sync but instead of of the src lane to pull
    // from being an absoute one it offset. So 1 is the same as lane + 1. So
    // if the current lane is 5 then the value in down will be from lane 6.
    // The 31 just gets is own value back.
    int down = __shfl_down_sync(0xffffffff, value, 1);

    // Warp reduction (sum of all 32 lanes' values) using only registers.
    // Notice that each thread will use the same offset but since each thread
    // has a different thread id the offsets will be different. Also notice that
    // we are pulling the `sum` from the other lanes. 
    // So all 32 threads will perform this operation using the offset.
    int sum = value;
    for (int offset = 16, i = 0; offset > 0; offset /= 2, ++i) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
	    printf("[%d] lane %2d: sum=%d\n", i, lane, sum);
	}
    }

    if (lane == 0) {
    	printf("lane %2d: value=%3d sum=%d\n", lane, value, sum);
    }
}

int main() {
    shuffle_examples<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
