#ifndef KERNEL
#define KERNEL

#define BLOCK_SIZE 1024
#include <cuda_runtime.h>

int UI(int argc, char* argv[], int* jkl);
void initData(unsigned int* data, int len);
void showData(unsigned int* data, int len);
void cumSumCPU(unsigned int* data, unsigned int* cumSum, int len);
extern "C" void checkCUDAError(cudaError_t e);
extern "C" void kernel(dim3 grid, dim3 threads, unsigned int* d_data, unsigned int* d_flags, unsigned int* d_scan_value, unsigned int* d_DCount, int dim);
void resultCheck(unsigned int* result_CPU, unsigned int* result_GPU, int size);
// void resultCheck(unsigned int* result_CPU, unsigned int* result_GPU, int size);

#endif // !KERNEL