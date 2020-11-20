#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// UI for main function
// return 0 means everything's fine, just continue;
// return 1 means there's invalid input or '--help', terminate running.
int UI(int argc, char* argv[], int* param) {

    //input -h for help
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

    // input -i with dim followed
    if (argc == 3 && (strcmp(argv[1], "-i") == 0 || strcmp(argv[1], "--input") == 0)) {
        int dim = atoi(argv[2]);

        // check if dim in range
        // if dim < 0 or dim > 2048*65535, then return 1.
        if (dim < 0 || dim > 2048 * 65535) {
            printf("Invalid input dimention. Dimention should be ranged 0 ~ 2048 * 65535");
            return 1;
        }

        // dim is in range
        else {
            printf("Bin numbers: %d\n", dim);
            param[0] = dim;
            return 0;
        }
    }

    // all other invalid inputs
    else {
        printf("Invalid command. Please check how to make valid command by '-h' or '--help'.\n");
        return 1;
    }
}

// initialize data with int type range [0, 10)
void initData(unsigned int* data, int len) {
    for (int i = 0; i < len; ++i)
        data[i] = rand() % 10;
    return;
}

// show the data in the command prompt.
// this function is used for configuration
// only show previous 10 elements when length of array is too large
void showData(unsigned int* data, int len) {
    printf("\n[");
    for (int i = 0; i < len && i < 10; ++i) {
        if (i != 0) printf(",");
        printf("%4u", data[i]);
    }
    if (len > 10) printf("...");
    printf("]\n");
    return;
}

void cumSumCPU(unsigned int* data, unsigned int* cumSum, int len) {
    unsigned int res = 0;
    for (int i = 0; i < len; ++i) {
        res += data[i];
        cumSum[i] = res;
    }
    return;
}

// check if two array is exactly the same
void resultCheck(unsigned int* result_CPU, unsigned int* result_GPU, int size) {
    for (int i = 0; i < size; ++i) {
        if (result_CPU[i] != result_GPU[i]) {
            printf("\nResult check: Error!!!! Didn't pass. Line %d.", i);
            return;
        }
    }
    printf("\nResult check: ---PASS---.");
    return;
}