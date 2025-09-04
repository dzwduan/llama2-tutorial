#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../solve.h"

int test_relu_example1() {
    printf("  Test 1:\n");
    
    int size = 5;
    float h_input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float h_output[5] = {0};
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_relu(d_input, d_output, size);
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    [%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("    Output:   [%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);
    printf("    Expected: [%.1f, %.1f, %.1f, %.1f, %.1f]\n", 
           expected[0], expected[1], expected[2], expected[3], expected[4]);
    
    int passed = 1;
    for (int i = 0; i < size; i++) {
        if (fabsf(h_output[i] - expected[i]) > 1e-6f) {
            passed = 0;
            break;
        }
    }
    
    cudaFree(d_input); cudaFree(d_output);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

int test_relu_example2() {
    printf("  Test 2:\n");
    
    int size = 3;
    float h_input[] = {-3.5f, 0.0f, 4.2f};
    float h_output[3] = {0};
    float expected[] = {0.0f, 0.0f, 4.2f};
    
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_relu(d_input, d_output, size);
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input:    [%.1f, %.1f, %.1f]\n", h_input[0], h_input[1], h_input[2]);
    printf("    Output:   [%.1f, %.1f, %.1f]\n", h_output[0], h_output[1], h_output[2]);
    printf("    Expected: [%.1f, %.1f, %.1f]\n", expected[0], expected[1], expected[2]);
    
    int passed = 1;
    for (int i = 0; i < size; i++) {
        if (fabsf(h_output[i] - expected[i]) > 1e-6f) {
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
    printf("CUDA ReLU Activation Tests\n");
    printf("--------------------------\n");
    
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
    if (test_relu_example1()) passed++;
    if (test_relu_example2()) passed++;
    
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