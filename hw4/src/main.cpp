#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "kernel.h"



int main(int argc, char* argv[]) {

    // reset rand seed
    srand((unsigned)time(NULL));
    clock_t start, finish;
    int total_time;

    // Go through UI first.
    // In UI section, only command with valid param can go to the next step.
    int UIStatus;
    int param[1];
    UIStatus = UI(argc, argv, param);
    if (UIStatus != 0) {
        printf("\nApplication terminates.\n");
        return 0;
    }
    // UI section ends

    // Initialize data array with unsigned int type
    const int dim = param[0];
    unsigned int* data = (unsigned int*)malloc(dim * sizeof(unsigned int));
    initData(data, dim);
    showData(data, dim);
    printf("Done initializing data array with length %d.\n", dim);
    // Initialzing ends

    // CPU code for calculating the cum-sum
    // Use this result to varify the kernel result later
    unsigned int* cumSum_CPU = (unsigned int*)malloc(dim * sizeof(unsigned int));
    unsigned int* cumSum_GPU = (unsigned int*)malloc(dim * sizeof(unsigned int));
    start = clock();
    cumSumCPU(data, cumSum_CPU, dim);
    finish = clock();
    total_time = (int)(finish - start);
    printf("\ncumSum_CPU:");
    showData(cumSum_CPU, dim);
    printf("Done histogrm calculation with CPU in %d miliseconds.\n", total_time);
    // Histogram calculating with CPU ends

    // Allocate device memory, copy data from host to device
    unsigned int* d_data, *d_flags, *d_scan_value, *d_DCount;
    checkCUDAError(cudaMalloc((unsigned int**)&d_data, dim * sizeof(unsigned int)));
    checkCUDAError(cudaMalloc((unsigned int**)&d_DCount, sizeof(unsigned int)));
    checkCUDAError(cudaMalloc((unsigned int**)&d_flags, (int)ceil(1.0 * dim / BLOCK_SIZE) * sizeof(unsigned int)));
    checkCUDAError(cudaMalloc((unsigned int**)&d_scan_value, (int)ceil(1.0 * dim / BLOCK_SIZE) * sizeof(unsigned int)));
    int dCount = 0;
    unsigned int flagOne = 1;
    // checkCUDAError(cudaMemcpyToSymbol(DCount, &dCount, sizeof(int)));
    // checkCUDAError(cudaMemcpyToSymbol(&d_flags[0], &flagOne, sizeof(unsigned int)));
    checkCUDAError(cudaMemset(d_flags, 0, (int)ceil(1.0 * dim / BLOCK_SIZE) * sizeof(unsigned int)));
    // checkCUDAError(cudaMemset(d_flags, 1, sizeof(unsigned int)));
    checkCUDAError(cudaMemset(d_DCount, 0, sizeof(unsigned int)));
    printf("\nDone allocating space in device.");
    checkCUDAError(cudaMemcpy(d_data, data, dim * sizeof(unsigned int), cudaMemcpyHostToDevice));
    printf("\nDone copying memory from host to device.");
    // Done allocating, transfering and initializing

    
    // Initialize thread block and kernel grid dimensions
    dim3 threads(BLOCK_SIZE);
    dim3 grid((int)ceil(1.0 * dim / threads.x));
    // dim3 grid(120);
    printf("\nDone initializing block dimention and grid dimention.");
    // Done initializing thread block and kernel grid dimensions

    
    // launch CUDA device kernel
    start = clock();
    kernel(grid, threads, d_data, d_flags, d_scan_value, d_DCount, dim);
    checkCUDAError(cudaDeviceSynchronize());
    finish = clock();
    total_time = (int)(finish - start);
    printf("\nDone cumSum with GPU in %d miliseconds.\n", total_time);
    // Done CUDA device kernel

    
    // Copy results from device to host and free device memory
    checkCUDAError(cudaMemcpy(cumSum_GPU, d_data, dim * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("hist_GPU:");
    showData(cumSum_GPU, dim);
    checkCUDAError(cudaFree(d_data));
    checkCUDAError(cudaFree(d_flags));
    checkCUDAError(cudaFree(d_scan_value));
    checkCUDAError(cudaFree(d_DCount));
    // Done copying results and freeing device memory

    
    // Check the result of the Calculated Matrix
    resultCheck(cumSum_CPU, cumSum_GPU, dim);
    // Done result checking.
    
    return 0;
}