#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../solve.h"

int test_reduction_example1() {
    printf("  Test 1:\n");
    
    int size = 8;
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_output = 0.0f;
    float expected = 36.0f;
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_reduction_sum(d_input, d_output, size);
    
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4], h_input[5], h_input[6], h_input[7]);
    printf("    Output:   %.1f\n", h_output);
    printf("    Expected: %.1f\n", expected);
    
    int passed = fabsf(h_output - expected) < 1e-6f;
    
    cudaFree(d_input); cudaFree(d_output);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

int test_reduction_example2() {
    printf("  Test 2:\n");
    
    int size = 4;
    float h_input[] = {-2.5f, 1.5f, -1.0f, 2.0f};
    float h_output = 0.0f;
    float expected = 0.0f;
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_reduction_sum(d_input, d_output, size);
    
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    [%.1f, %.1f, %.1f, %.1f]\n", h_input[0], h_input[1], h_input[2], h_input[3]);
    printf("    Output:   %.1f\n", h_output);
    printf("    Expected: %.1f\n", expected);
    
    int passed = fabsf(h_output - expected) < 1e-6f;
    
    cudaFree(d_input); cudaFree(d_output);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

int test_reduction_large() {
    printf("  Test 3:\n");
    
    int size = 10000;
    float *h_input = (float*)malloc(size * sizeof(float));
    float h_output = 0.0f;
    float expected = (float)size * (size + 1) / 2.0f;
    
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(i + 1);
    }
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_reduction_sum(d_input, d_output, size);
    
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    Large array (10000 elements: 1 to 10000)\n");
    printf("    Output:   %.0f\n", h_output);
    printf("    Expected: %.0f\n", expected);
    
    int passed = fabsf(h_output - expected) < 1.0f;  // allow larger error due to floating point precision
    
    cudaFree(d_input); cudaFree(d_output);
    free(h_input);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Conditional compilation: only include main function when testing standalone
#ifdef STANDALONE_TEST
int main() {
    printf("CUDA Reduction Tests\n");
    printf("--------------------\n");
    
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
    if (test_reduction_example1()) passed++;
    if (test_reduction_example2()) passed++;
    if (test_reduction_large()) passed++;
    
    printf("\nResults: %d/3 tests passed\n", passed);
    
    if (passed == 3) {
        printf("All tests passed\n");
        return 0;
    } else {
        printf("Some tests failed\n");
        return 1;
    }
}
#endif