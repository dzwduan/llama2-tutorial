#ifndef SOLVE_H
#define SOLVE_H

void solve_matrix_mult(const float* A, const float* B, float* C, int M, int N, int K);
void solve_matrix_transpose(const float* input, float* output, int rows, int cols);
void solve_relu(const float* input, float* output, int size);
void solve_leaky_relu(const float* input, float* output, int N);
void solve_reduction_sum(const float* input, float* output, int size);
void solve_softmax(const float* input, float* output, int size);

#endif // SOLVE_H