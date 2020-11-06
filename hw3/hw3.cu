#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE_WIDTH 16
#define MAX_MASK_WIDTH 128

int UI(int argc, char* argv[], int* jkl);
void initData(int* data, int x, int y);
void initMask(float* mask, int k);
void showData(int* data, int len);
void showMask(float* mask, int len);
void convCPU(int* data, float* mask, float* conv_CPU, int x, int y, int k);
void checkCUDAError(cudaError_t e);
__global__
void convGPU(int* d_data, float* d_conv, int x, int y, int k);
void resultCheck(float* result_CPU, float* result_GPU, int size);
__constant__ float M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

int main(int argc, char* argv[]){
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    printf("Total constant memory: %zd \n", dev_prop.totalConstMem);
    
    // reset rand seed
    srand((unsigned)time(NULL));
    clock_t start, finish;
    int total_time;

    // Go through UI first.
    // In UI section, only command with valid param can go to the next step.
    int UIStatus;
    int param[3];
    UIStatus = UI(argc, argv, param);
    if (UIStatus != 0) {
        printf("\nApplication terminates.\n");
        return 0;
    }
    // UI section ends

    
    // Initialize data array with int type
    const int x = param[0];
    const int y = param[1];
    const int k = param[2];
    int* data = (int*)malloc(x * y * sizeof(int));
    float* mask = (float*)malloc(k * k * sizeof(float));
    initData(data, x, y);
    initMask(mask, k);
    showData(data, x * y);
    showMask(mask, k * k);
    printf("Done initializing data array.\n");
    // Initialzing ends

    // CPU code for calculating convolution
    // Use this result to varify the kernel result later
    float* conv_CPU = (float*)calloc(x * y, sizeof(float));
    float* conv_GPU = (float*)calloc(x * y, sizeof(float));
    start = clock();
    convCPU(data, mask, conv_CPU, x, y, k);
    finish = clock();
    total_time = (int)(finish - start);
    showMask(conv_CPU, x * y);
    printf("Done convolution with CPU in %d miliseconds.\n", total_time);
    // Convolution calculating with CPU ends

    
    // Allocate device memory, copy data from host to device
    int* d_data; 
    float *d_conv;
    checkCUDAError(cudaMalloc((int**)&d_data, x * y * sizeof(int)));
    checkCUDAError(cudaMalloc((float**)&d_conv, x * y * sizeof(float)));
    printf("Done allocating space in device.");
    checkCUDAError(cudaMemcpy(d_data, data, x * y * sizeof(int), cudaMemcpyHostToDevice));
    checkCUDAError(cudaMemcpyToSymbol(M, mask, k * k * sizeof(float)));
    printf("\nDone copying memory from host to device.");
    // Done allocating, transfering and initializing

    // Initialize thread block and kernel grid dimensions
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((int)ceil(1.0 * x / threads.x), (int)ceil(1.0 * y / threads.y));
    printf("\nDone initializing block dimention and grid dimention.");
    // Done initializing thread block and kernel grid dimensions
    
    // launch CUDA device kernel
    start = clock();
    convGPU<<< grid, threads >>>(d_data, d_conv, x, y, k);
    checkCUDAError(cudaDeviceSynchronize());
    finish = clock();
    total_time = (int)(finish - start);
    printf("\nDone matrix multiplication with GPU in %d miliseconds.\n", total_time);
    // Done CUDA device kernel

    
    // Copy results from device to host and free device memory
    checkCUDAError(cudaMemcpy(conv_GPU, d_conv, x * y * sizeof(float), cudaMemcpyDeviceToHost));
    printf("conv_GPU:");
    showMask(conv_GPU, x * y);
    checkCUDAError(cudaFree(d_conv));
    checkCUDAError(cudaFree(d_data));
    // Done copying results and freeing device memory

    
    // Check the result of the Calculated Matrix
    resultCheck(conv_CPU, conv_GPU, x * y);
    // Done result checking.
    
    return 0;
}

// UI for main function
// return 0 means everything's fine, just continue;
// return 1 means there's invalid input or '--help', terminate running.
int UI(int argc, char* argv[], int* param) {
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printf("CUDA Programming Homework. Histogram Algorithm.\n");
        printf("\nUsage: hist [OPTION]...\n");
        printf("\nOptions:\n");
        printf("%5s, %-10s %-50s\n", "-h", "--help", "Show helping information.");
        printf("%5s, %-10s %-50s\n", "-i", "--input", "Followed by 3 integers as input parameters.");
        printf("\nExamples:\n");
        printf("hist -h\n");
        printf("  Shows the helping information.\n");
        printf("hist -i 1960 1080 9\n");
        printf("  1960 1080 represents the picture is 1960 * 1080, 9 means the 2D mask is 9 * 9\n");
        return 1;
    }

    if (argc == 5 && (strcmp(argv[1], "-i") == 0 || strcmp(argv[1], "--input") == 0)) {
        int x = atoi(argv[2]);
        int y = atoi(argv[3]);
        int k = atoi(argv[4]);
        if (x <= 0 || y <= 0 || k <= 0) {
            printf("Invalid array length. The input values should be an integer greater than 0.\n");
            return 1;
        }
        if (k % 2 == 0) {
            printf("Invalid k, k should be odd.\n");
            return 1;
        }
        if (k > MAX_MASK_WIDTH) {
            printf("Invalid k, k is too big. Can't store it in texture memory.\n");
            return 1;
        }
        else {
            printf("x: %d\n", x);
            printf("y: %d\n", y);
            printf("k: %d\n", k);
            param[0] = x;
            param[1] = y;
            param[2] = k;
            return 0;
        }
    }
    else {
        printf("Invalid command. Please check how to make valid command by '-h' or '--help'.\n");
        return 1;
    }
}

void initData(int* data, int x, int y) {
    for (int i = 0; i < x * y; ++i) 
        data[i] = rand() % 16;
    return;
}

void initMask(float* mask, int k) {
    float sum = 0.0f;
    for (int i = 0; i < k * k; ++i) {
        mask[i] = rand() % 100 / 100.0f;
        sum += mask[i];
    }
    for (int i = 0; i < k * k; ++i) {
        mask[i] /= sum;
    }
    return;
}

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

void showMask(float* mask, int len) {
    printf("data:\n[");
    for (int i = 0; i < len && i < 10; ++i) {
        if (i != 0) printf(",");
        printf("%4f", mask[i]);
    }
    if (len > 10) printf("...");
    printf("]\n");
    return;
}

void convCPU(int* data, float* mask, float* conv_CPU, int x, int y, int k) {
    for (int i = 0; i < y; ++i) {
        for (int j = 0; j < x; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < k; ++m) {
                for (int n = 0; n < k; ++n) {
                    if (i + m - k / 2 >= 0 && i + m - k / 2 < y && j + n - k / 2 >= 0 && j + n - k / 2 < x) {
                        sum += data[(i + m - k / 2) * x + (j + n - k / 2)] * mask[m * k + n];
                    }
                }
            }
            conv_CPU[i * x + j] = sum;
        }
    }
    return;
}

void checkCUDAError(cudaError_t e) {
    if (e == 0) return;
    printf("\nError: %s\n", cudaGetErrorName(e));
    printf("%s\n", cudaGetErrorString(e));
    exit(0);
}

__global__
void convGPU(int* d_data, float* d_conv, int x, int y, int k) {
    unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float data_ds[TILE_WIDTH][TILE_WIDTH];
    if (ty < y && tx < x) {
        data_ds[threadIdx.y][threadIdx.x] = d_data[ty * x + tx];
        __syncthreads();
        float Pvalue = 0;
        for (int m = 0; m < k; ++m) {
            for (int n = 0; n < k; ++n) {
                int cur_row = ty + m - k / 2;
                int cur_col = tx + n - k / 2;
                if (cur_col >= 0 && cur_col < x && cur_row >= 0 && cur_row < y) {
                    if (cur_col >= blockDim.x * blockIdx.x && cur_col < blockDim.x * (blockIdx.x + 1)
                        && cur_row >= blockDim.y * blockIdx.y && cur_row < blockDim.y * (blockIdx.y + 1))
                        Pvalue += data_ds[threadIdx.y + m - k / 2][threadIdx.x + n - k / 2] * M[m * k + n];
                    else
                        Pvalue += d_data[cur_row * x + cur_col] * M[m * k + n];
                }
            }
        }
        d_conv[ty * x + tx] = Pvalue;
    }
}

void resultCheck(float* result_CPU, float* result_GPU, int size) {
    for (int i = 0; i < size; ++i) {
        if (result_CPU[i] * 1.001 <= result_GPU[i] || result_CPU[i] * 0.999 >= result_GPU[i]) {
            printf("\nResult check: Error!!!! Didn't pass.");
            return;
        }
    }
    printf("\nResult check: ---PASS---.");
    return;
}
