#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../solve.h"

int test_matrix_mult_2x2() {
    printf("  Test 1:\n");
    
    int M = 2, N = 2, K = 2;
    float h_A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C[4] = {0};
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_matrix_mult(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input A:  [[%.1f, %.1f], [%.1f, %.1f]]\n", h_A[0], h_A[1], h_A[2], h_A[3]);
    printf("    Input B:  [[%.1f, %.1f], [%.1f, %.1f]]\n", h_B[0], h_B[1], h_B[2], h_B[3]);
    printf("    Output:   [[%.1f, %.1f], [%.1f, %.1f]]\n", h_C[0], h_C[1], h_C[2], h_C[3]);
    printf("    Expected: [[%.1f, %.1f], [%.1f, %.1f]]\n", expected[0], expected[1], expected[2], expected[3]);
    
    int passed = 1;
    for (int i = 0; i < M * K; i++) {
        if (fabsf(h_C[i] - expected[i]) > 1e-5f) {
            passed = 0;
            break;
        }
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

int test_matrix_mult_1x3() {
    printf("  Test 2:\n");
    
    int M = 1, N = 3, K = 1;
    float h_A[] = {1.0f, 2.0f, 3.0f};
    float h_B[] = {4.0f, 5.0f, 6.0f};
    float h_C[1] = {0};
    float expected = 32.0f;
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    
    solve_matrix_mult(d_A, d_B, d_C, M, N, K);
    
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("    Input A:  [%.1f, %.1f, %.1f]\n", h_A[0], h_A[1], h_A[2]);
    printf("    Input B:  [%.1f, %.1f, %.1f]\n", h_B[0], h_B[1], h_B[2]);
    printf("    Output:   [%.1f]\n", h_C[0]);
    printf("    Expected: [%.1f]\n", expected);
    
    int passed = fabsf(h_C[0] - expected) < 1e-5f;
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    printf("    %s\n", passed ? "PASS" : "FAIL");
    return passed;
}

// Conditional compilation: only include main function when testing standalone
#ifdef STANDALONE_TEST
int main() {
    printf("CUDA Matrix Multiplication Tests\n");
    printf("---------------------------------\n");
    
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
    if (test_matrix_mult_2x2()) passed++;
    if (test_matrix_mult_1x3()) passed++;
    
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