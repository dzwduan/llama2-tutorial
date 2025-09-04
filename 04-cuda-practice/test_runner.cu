#include <stdio.h>
#include <cuda_runtime.h>
#include "solve.h"

// Matrix Multiplication tests
extern int test_matrix_mult_2x2();
extern int test_matrix_mult_1x3();

// Matrix Transpose tests
extern int test_transpose_example1();
extern int test_transpose_example2();

// ReLU Activation tests
extern int test_relu_example1();
extern int test_relu_example2();

// Leaky ReLU tests
extern int test_leaky_relu_example1();
extern int test_leaky_relu_example2();

// Reduction tests
extern int test_reduction_example1();
extern int test_reduction_example2();
extern int test_reduction_large();

// Softmax tests
extern int test_softmax_example1();
extern int test_softmax_example2();

int main() {
    printf("CUDA Tests\n");
    
    int total_passed = 0;
    int total_tests = 0;
    
    printf("\n=== Matrix Multiplication ===\n");
    total_tests++; if (test_matrix_mult_2x2()) total_passed++;
    total_tests++; if (test_matrix_mult_1x3()) total_passed++;
    
    printf("\n=== Matrix Transpose ===\n");
    total_tests++; if (test_transpose_example1()) total_passed++;
    total_tests++; if (test_transpose_example2()) total_passed++;
    
    printf("\n=== ReLU Activation ===\n");
    total_tests++; if (test_relu_example1()) total_passed++;
    total_tests++; if (test_relu_example2()) total_passed++;
    
    printf("\n=== Leaky ReLU ===\n");
    total_tests++; if (test_leaky_relu_example1()) total_passed++;
    total_tests++; if (test_leaky_relu_example2()) total_passed++;
    
    printf("\n=== Reduction ===\n");
    total_tests++; if (test_reduction_example1()) total_passed++;
    total_tests++; if (test_reduction_example2()) total_passed++;
    total_tests++; if (test_reduction_large()) total_passed++;
    
    printf("\n=== Softmax ===\n");
    total_tests++; if (test_softmax_example1()) total_passed++;
    total_tests++; if (test_softmax_example2()) total_passed++;
    
    printf("\nResults: %d/%d tests passed\n", total_passed, total_tests);
    
    if (total_passed == total_tests) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}