#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../solve.h"

int test_transpose_example1() {
    printf("  Test 1:\n");
    
    int rows = 2, cols = 3;
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_output[6] = {0};
    float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));
    
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_matrix_transpose(d_input, d_output, rows, cols);
    
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    [[%.1f, %.1f, %.1f], [%.1f, %.1f, %.1f]]\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4], h_input[5]);
    printf("    Output:   [[%.1f, %.1f], [%.1f, %.1f], [%.1f, %.1f]]\n", 
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4], h_output[5]);
    printf("    Expected: [[%.1f, %.1f], [%.1f, %.1f], [%.1f, %.1f]]\n", 
           expected[0], expected[1], expected[2], expected[3], expected[4], expected[5]);
    
    int passed = 1;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(h_output[i] - expected[i]) > 1e-5f) {
            passed = 0;
            break;
        }
    }
    
    cudaFree(d_input); cudaFree(d_output);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

int test_transpose_example2() {
    printf("  Test 2:\n");
    
    int rows = 3, cols = 1;
    float h_input[] = {1.0f, 2.0f, 3.0f};
    float h_output[3] = {0};
    float expected[] = {1.0f, 2.0f, 3.0f};
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));
    
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_matrix_transpose(d_input, d_output, rows, cols);
    
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    [[%.1f], [%.1f], [%.1f]]\n", h_input[0], h_input[1], h_input[2]);
    printf("    Output:   [%.1f, %.1f, %.1f]\n", h_output[0], h_output[1], h_output[2]);
    printf("    Expected: [%.1f, %.1f, %.1f]\n", expected[0], expected[1], expected[2]);
    
    int passed = 1;
    for (int i = 0; i < rows * cols; i++) {
        if (fabsf(h_output[i] - expected[i]) > 1e-5f) {
            passed = 0;
            break;
        }
    }
    
    cudaFree(d_input); cudaFree(d_output);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Conditional compilation: only include main function when testing standalone
#ifdef STANDALONE_TEST
int main() {
    printf("CUDA Matrix Transpose Tests\n");
    printf("---------------------------\n");
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);
    
    printf("Running tests...\n");
    
    int passed = 0;
    if (test_transpose_example1()) passed++;
    if (test_transpose_example2()) passed++;
    
    printf("\nResults: %d/2 tests passed\n", passed);
    
    if (passed == 2) {
        printf("All tests passed\n");
        return 0;
    } else {
        printf("Some tests failed\n");
        return 1;
    }
}
#endif