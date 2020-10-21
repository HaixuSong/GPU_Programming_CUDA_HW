#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256

int UI(int argc, char* argv[], int* jkl);
void initData(int* data, int len);
void showData(int* data, int len);
void histCountCPU(int* data, int* hist, int len, int bin);
void checkCUDAError(cudaError_t e);
__global__
void histCountGPU(int* d_data, int* d_hist, int len, int bin, int part);
void resultCheck(int* result_CPU, int* result_GPU, int size);

int main(int argc, char* argv[]){

    
    // reset rand seed
    srand((unsigned)time(NULL));
    clock_t start, finish;
    int total_time;

    // Go through UI first.
    // In UI section, only command with valid param can go to the next step.
    int UIStatus;
    int param[2];
    UIStatus = UI(argc, argv, param);
    if (UIStatus != 0) {
        printf("\nApplication terminates.\n");
        return 0;
    }
    // UI section ends

    // Initialize data array with int type
    const int bin = param[0];
    const int len = param[1];
    int* data = (int*)malloc(len * sizeof(int));
    initData(data, len);
    showData(data, len);
    printf("Done initializing data array with length %d.\n", len);
    // Initialzing ends

    
    // CPU code for calculating the histogram
    // Use this result to varify the kernel result later
    // Using calloc to initialize the hist with zeros.
    int* hist_CPU = (int*)calloc(len, sizeof(int));
    int* hist_GPU = (int*)calloc(len, sizeof(int));
    start = clock();
    histCountCPU(data, hist_CPU, len, bin);
    finish = clock();
    total_time = (int)(finish - start);
    printf("\nhist_CPU:");
    showData(hist_CPU, bin);
    printf("Done histogrm calculation with CPU in %d miliseconds.\n", total_time);
    // Histogram calculating with CPU ends

    // Allocate device memory, copy data from host to device, initialize device histogram with zeros.
    int *d_data, *d_hist;
    checkCUDAError(cudaMalloc((int**)&d_data, len * sizeof(int)));
    checkCUDAError(cudaMalloc((int**)&d_hist, bin * sizeof(int)));
    printf("\nDone allocating space in device.");
    checkCUDAError(cudaMemcpy(d_data, data, len * sizeof(int), cudaMemcpyHostToDevice));
    printf("\nDone copying memory from host to device.");
    checkCUDAError(cudaMemset(d_hist, 0, bin));
    printf("\nDone initializing device histogram with zeros.\n");
    // Done allocating, transfering and initializing

    
    // Initialize thread block and kernel grid dimensions
    dim3 threads(BLOCK_SIZE);
    // dim3 grid((int)ceil(1.0 * len / threads.x));
    dim3 grid(120);
    printf("\nDone initializing block dimention and grid dimention.");
    // Done initializing thread block and kernel grid dimensions
    
    // launch CUDA device kernel
    start = clock();
    histCountGPU<<< grid, threads, bin * sizeof(int) >>>(d_data, d_hist, len, bin, 1024 / bin);
    // Done CUDA device kernel

    // Copy results from device to host and free device memory
    checkCUDAError(cudaDeviceSynchronize());
    finish = clock();
    total_time = (int)(finish - start);
    printf("\nDone matrix multiplication with GPU in %d miliseconds.\n", total_time);
    checkCUDAError(cudaMemcpy(hist_GPU, d_hist, bin * sizeof(int), cudaMemcpyDeviceToHost));
    printf("hist_GPU:");
    showData(hist_GPU, bin);
    checkCUDAError(cudaFree(d_hist));
    checkCUDAError(cudaFree(d_data));
    // Done copying results and freeing device memory

    // Check the result of the Calculated Matrix
    resultCheck(hist_CPU, hist_GPU, bin);
    // Done result checking.

    return 0;
}

// UI for main function
// return 0 means everything's fine, just continue;
// return 1 means there's invalid input or '--help', terminate running.
int UI(int argc, char* argv[], int* param) {
    // UI for the exe file
    // while input with -h or --help; tell that what we need as params in linux style
    // while input with 0 or 1 or more than 2 parameters; tell that we need 2 params
    // while input with 2 paramerters; print the size of two input matrix; check if all params are valid;
    // param[0] is valid if it is exponent of 2. param[1] is valid if it is greater than 0
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("CUDA Programming Homework. Histogram Algorithm.\n");
        printf("\nUsage: hist [OPTION]...\n");
        printf("\nOptions:\n");
        printf("%5s, %-10s %-50s\n", "-h", "--help", "Show helping information.");
        printf("%5s, %-10s %-50s\n", "-i", "--input", "Followed by 2 integers as input parameters.");
        printf("\nExamples:\n");
        printf("hist -h\n");
        printf("  Shows the helping information.\n");
        printf("hist -i 8 200\n");
        printf("  8 represents 8 bins in histogram, 200 means the length of the data\n");
        return 1;
    }

    if (argc == 4 && (strcmp(argv[1], "-i") == 0 || strcmp(argv[1], "--input") == 0)) {
        int bin = atoi(argv[2]);
        int len = atoi(argv[3]);
        int div, mod, cache = bin, count = 0;
        while (cache > 1) {
            ++count;
            div = cache / 2;
            mod = cache - div * 2;
            if (mod == 1) {
                printf("Invalid bin numbers. The bin numbers should be exponent of 2, range from 2^2 to 2^8\n");
                return 1;
            }
            cache = div;
        }
        if (count > 8 || count < 2) {
            printf("Invalid bin numbers. The bin numbers should be exponent of 2, range from 2^2 to 2^8\n");
            return 1;
        }
        if (len <= 0) {
            printf("Invalid array length. The array length should be an integer greater than 0.\n");
            return 1;
        }
        else {
            printf("Bin numbers: %d\n", bin);
            printf("Array length: %d\n", len);
            param[0] = bin;
            param[1] = len;
            return 0;
        }
    }

    else {
        printf("Invalid command. Please check how to make valid command by '-h' or '--help'.\n");
        return 1;
    }
}

// initialize data with int type range [0, 1024)
void initData(int* data, int len) {
    for (int i = 0; i < len; ++i) 
        data[i] = rand() % 1024;
    return;
}

// show the data in the command prompt.
// this function is used for configuration
// only show previous 10 elements when length of array is too large
void showData(int* data, int len) {
    printf("data:\n[");
    for (int i = 0; i < len && i < 10; ++i) {
        if (i != 0) printf(",");
        printf("%4d", data[i]);
    }
    if (len > 10) printf("...");
    printf("]\n");
    return;
}

// matrix multiplication with CPU in the most stupid algo
// Algo Complexity: O((2k-1)*j*l)
void histCountCPU(int* data, int* hist, int len, int bin) {
    int part = 1024 / bin;
    for (int i = 0; i < len; ++i) {
        ++hist[data[i] / part];
    }
    return;
}

// Check the result of the CUDA function.
void checkCUDAError(cudaError_t e) {
    if (e == 0) return;
    printf("\nError: %s\n", cudaGetErrorName(e));
    printf("%s\n", cudaGetErrorString(e));
    exit(0);
}

// matrix multiplication with GPU device
// using tiling algorithm, take use of the shared memory to higher the compute-to-global-memory-access ratio
__global__
void histCountGPU(int* d_data, int* d_hist, int len, int bin, int part) {
    // calculating thread id
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Privatized bins
    extern __shared__ int histo_s[];
    for (unsigned int binIdx = threadIdx.x; binIdx < bin; binIdx += blockDim.x) {
        histo_s[binIdx] = 0;
    }
    __syncthreads();

    // Histogram Count
    for (unsigned int i = tid; i < len; i += blockDim.x * gridDim.x) {
        atomicAdd(&(histo_s[d_data[i] / part]), 1);
    }
    __syncthreads();

    // Commit to global memory
    for (unsigned int binIdx = threadIdx.x; binIdx < bin; binIdx += blockDim.x) {
        atomicAdd(&(d_hist[binIdx]), histo_s[binIdx]);
    }
}

// check if two array is exactly the same
void resultCheck(int* result_CPU, int* result_GPU, int size) {
    for (int i = 0; i < size; ++i) {
        if (result_CPU[i] != result_GPU[i]) {
            printf("\nResult check: Error!!!! Didn't pass.");
            return;
        }
    }
    printf("\nResult check: ---PASS---.");
    return;
}
