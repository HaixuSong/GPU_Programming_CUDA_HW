#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ELEMENT_MIN 0
#define ELEMENT_MAX 10
#define BLOCK_SIZE 16
#define TILE_SIZE 16
#define ZERO 1.e-6

int UI(int argc, char* argv[], int* jkl);
float randGenerate(int min, int max);
void initMatrix(float* matrix, int size, int min, int max);
void matrixMulCPU(float* A, float* B, float* C, int j, int k, int l);
__global__
void matrixMulGPU(float* d_A, float* d_B, float* d_C, int j, int k, int l);
void showMatrix(float* matrix, int row, int col);
void resultCheck(float* result_CPU, float* result_GPU, int size);

int main(int argc, char* argv[]){

    // reset rand seed
    srand((unsigned)time(NULL));
    clock_t start, finish;
    int total_time;

    // Go through UI first.
    // In UI section, only command with valid param can go to the next step.
    int UIStatus;
    int jkl[3];
    UIStatus = UI(argc, argv, jkl);
    if (UIStatus != 0) {
        printf("\nApplication terminates.");
        return 0;
    }
    printf("\nContinuing with j=%d, k=%d, l=%d", jkl[0], jkl[1], jkl[2]);
    // UI section ends

    // Initialize these two matrix A, B with random float type
    const int j = jkl[0];
    const int k = jkl[1];
    const int l = jkl[2];
    float* A = (float*)malloc(j * k * sizeof(float));
    float* B = (float*)malloc(k * l * sizeof(float));
    initMatrix(A, (j * k), ELEMENT_MIN, ELEMENT_MAX);
    initMatrix(B, (k * l), ELEMENT_MIN, ELEMENT_MAX);
    showMatrix(A, j, k);
    showMatrix(B, k, l);
    printf("\nDone initializing matrix A with size %d*%d and matrix B with %d*%d", j, k, k, l);
    // Initialzing ends

    // CPU code for calculating the matrix multiplication
    // Use this result to varify the kernel result later
    // Only do this step when j, k, l is below 5000
    float* C_GPU = (float*)malloc(j * l * sizeof(float));
    float* C_CPU = (float*)malloc(j * l * sizeof(float));
    if (k < 5000 && j < 5000 && l < 5000) {
        start = clock();
        matrixMulCPU(A, B, C_CPU, j, k, l);
        finish = clock();
        printf("\nDone matrix multiplication with CPU");
        total_time = (int)(finish - start);
        printf("\n%d microseconds used.\n", total_time);
        showMatrix(C_CPU, j, l);
    }
    // Matrix multipication with CPU ends

    // Allocate device memory and copy data from host to device
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, j * k * sizeof(float));
    cudaMalloc((float**)&d_B, k * l * sizeof(float));
    cudaMalloc((float**)&d_C, j * l * sizeof(float));
    printf("\nDone allocating space in device.");
    cudaMemcpy(d_A, A, j * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * l * sizeof(float), cudaMemcpyHostToDevice);
    printf("\nDone copying memory from host to device");
    // Done allocating and transfering

    // Initialize thread block and kernel grid dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((int)ceil(1.0 * l / threads.x), (int)ceil(1.0 * j / threads.y));
    printf("\nDone initializing block dimention and grid dimention.");
    // Done initializing thread block and kernel grid dimensions

    // launch CUDA device kernel
    start = clock();
    matrixMulGPU<<< grid, threads>>>(d_A, d_B, d_C, j, k, l);
    // Done CUDA device kernel

    // Copy results from device to host and free device memory
    cudaMemcpy(C_GPU, d_C, j * l * sizeof(float), cudaMemcpyDeviceToHost);
    finish = clock();
    printf("\nDone matrix multiplication with GPU.");
    total_time = (int)(finish - start);
    printf("\n%d microseconds used.\n", total_time);
    showMatrix(C_GPU, j, l);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Done copying results and freeing device memory

    // Check the result of the Calculated Matrix
    if (k < 5000 && j < 5000 && l < 5000) resultCheck(C_CPU, C_GPU, j * l);
    // Done result checking.

    return 0;
}

// UI for main function
// return 0 means everything's fine, just continue;
// return 1 means there's invalid input or '--help', terminate running.
int UI(int argc, char* argv[], int* jkl) {
    // UI for the exe file
    // while input with ? or -h or --help; tell that what we need as params
    // while input with [1,2] || [4, +inf) parameters; tell that we need 3 params
    // while input with 3 paramerters; print the size of two input matrix; check if all params are valid;
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "?") == 0)) {
        printf("\nWe need 3 parameters j,k,l which represents the two matrix's dimention.\n");
        printf("The first one with size j*k, and the second one's is k*l.\n");
        printf("j,k,l must be int type and greater than 0.\n");
        return 1;
    }
    if (argc <= 3 || argc >= 5) {
        printf("\nError.\nWe need 3 parameters as input.\nYou can use '--help' for more details.\n");
        return 1;
    }
    if (argc == 4) {
        int n;
        printf("\nNumber Of Arguments Passed: %d", argc);
        printf("\n----Following Are The Command Line Arguments Passed----");
        for (int counter = 1; counter < argc; counter++) {
            n = atoi(argv[counter]);
            if (n > 0) {
                printf("\nargv[%d]: %d", counter, n);
                jkl[counter - 1] = n;
            }
            else {
                printf("\nargv[%d] is invalid", counter);
                return 1;
            }
        }
    }
    return 0;
}

// random float generator within min and max
float randGenerate(int min, int max) {
    float res;
    int r = rand();
    res = 1.0 * rand() / RAND_MAX * (max-min) + min;
    return res;
}

// initialize matrix with float type
void initMatrix(float* matrix, int size, int min, int max) {
    for (int i = 0; i < size; ++i) 
        matrix[i] = randGenerate(min, max);
    return;
}

// matrix multiplication with CPU in the most stupid algo
// Algo Complexity: O((2k-1)*j*l)
void matrixMulCPU(float* A, float* B, float* C, int j, int k, int l) {
    for (int row = 0; row < j; ++row) {
        for (int col = 0; col < l; ++col) {
            float eleSum = 0.0f;
            for (int i = 0; i < k; ++i) {
                eleSum += A[row * k + i] * B[i * l + col];
            }
            C[row * l + col] = eleSum;
        }
    }
}

// matrix multiplication with GPU device
// using tiling algorithm, take use of the shared memory to higher the compute-to-global-memory-access ratio
__global__
void matrixMulGPU(float* d_A, float* d_B, float* d_C, int j, int k, int l) {
    __shared__ float Ads[TILE_SIZE][TILE_SIZE];
    __shared__ float Bds[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float CEleValue = 0.0f;
    for (int ph = 0; ph * TILE_SIZE < k; ++ph) {
        if (row < j && ph * TILE_SIZE + tx < k)
            Ads[ty][tx] = d_A[row * k + ph * TILE_SIZE + tx];
        else
            Ads[ty][tx] = 0;
        if (col < l && ph * TILE_SIZE + ty < k)
            Bds[ty][tx] = d_B[(ph * TILE_SIZE + ty) * l + col];
        else
            Bds[ty][tx] = 0;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) {
            CEleValue += Ads[ty][i] * Bds[i][tx];
        }
        __syncthreads();
    }
    if (row < j && col < l)
        d_C[row * l + col] = CEleValue;
}

// show the data in the command prompt.
// this function is used for configuration
// please don't use it when j,k,l is too large
void showMatrix(float* matrix, int row, int col) {
    if (row > 5 || col > 5) return;
    printf("\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f\t", matrix[i*col + j]);
        }
        printf("\n");
    }
    return;
}

// check if two matrix is the same
// 0 is defined at the very begining
void resultCheck(float* result_CPU, float* result_GPU, int size) {
    // using relative error
    // assuming CPU result is the true value
    for (int i = 0; i < size; ++i) {
        if ((result_CPU[i] - result_GPU[i])/ result_CPU[i] > ZERO || (result_GPU[i] - result_CPU[i])/result_CPU[i] < -ZERO) {
            printf("\nResult check: Error!!!! Didn't pass.");
            return;
        }
    }
    printf("\nResult check: ---PASS---.");
    return;
}