#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernel.h"

/*
__device__
static void reductionTree(unsigned int* d_data) {

}

__device__
static void distributionTree(unsigned int* d_data) {

}

__device__
static void BKScanAlgo(unsigned int* d_data) {
    reductionTree(d_data);
    distributionTree(d_data);
}
*/

// matrix multiplication with GPU device
// using tiling algorithm, take use of the shared memory to higher the compute-to-global-memory-access ratio
__global__
static void cumSumGPU(unsigned int* d_data, unsigned int* flags, unsigned int* scan_value, unsigned int* d_DCount, int dim) {
    // Calculate bid due to which block is firstly loaded
    extern __shared__ unsigned int bid;
    if (threadIdx.x == 0) {
         bid = atomicAdd(&d_DCount[0], 1);
    }
    __syncthreads();

    extern __shared__ unsigned int partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * bid * blockDim.x;
    if (start + t < dim) partialSum[t] = d_data[start + t];
    if (start + t + blockDim.x < dim) partialSum[blockDim.x + t] = d_data[start + blockDim.x + t];
    __syncthreads();

    // Phase One
    // BKScanAlgo

    // Reduction Tree
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * blockDim.x) partialSum[index] += partialSum[index - stride];
        __syncthreads();
    }
    // Distribute Tree
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * blockDim.x) {
            partialSum[index + stride] += partialSum[index];
        }
    }
    __syncthreads();


    // Phase Two
    if (threadIdx.x == 0) {
        if (bid == 0) {
            scan_value[bid] = partialSum[2*blockDim.x - 1];
            atomicAdd(&flags[bid], 1);
        }
        else {
            // Wait for the previous flag
            while (atomicAdd(&flags[bid - 1], 0) == 0) { ; }
            // Read previous partial sum
            unsigned int previous_sum = scan_value[bid - 1];
            // Propagate partial sum
            scan_value[bid] = scan_value[bid - 1] + partialSum[2 * blockDim.x - 1];
            // Memory fence
            __threadfence();
            // Set flag
            atomicAdd(&flags[bid], 1);
        }
    }
    __syncthreads();

    // Phase Three
    if (bid > 0) {
        partialSum[t] += scan_value[bid - 1];
        partialSum[t + blockDim.x] += scan_value[bid - 1];
    }
    __syncthreads();

    if (start + t < dim)  d_data[start + t] = partialSum[t];
    if (start + t + blockDim.x < dim) d_data[start + blockDim.x + t] = partialSum[blockDim.x + t];
}

void kernel(dim3 grid, dim3 threads, unsigned int* d_data, unsigned int* d_flags, unsigned int* d_scan_value
    , unsigned int* d_DCount, int dim) {
    cumSumGPU <<<grid, threads, (2 * BLOCK_SIZE + 1) * sizeof(unsigned int) >>> (d_data, d_flags, d_scan_value, d_DCount, dim);
}

// Check the result of the CUDA function.
void checkCUDAError(cudaError_t e) {
    if (e == 0) return;
    printf("\nError: %s\n", cudaGetErrorName(e));
    printf("%s\n", cudaGetErrorString(e));
    exit(0);
}