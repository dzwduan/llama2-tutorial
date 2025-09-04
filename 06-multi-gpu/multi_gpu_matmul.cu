#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define MOD 1000000007

#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));               \
        MPI_Abort(MPI_COMM_WORLD, 1);                                       \
    }                                                                       \
} while (0)

__global__ void add_matrices_kernel(long* C, const long* A, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = (C[idx] + A[idx]) % MOD;
    }
}

__global__ void matmul_kernel(const long* A, const long* B, 
                                       long* C, int M, int N, int K) {
    __shared__ long As[16][16];
    __shared__ long Bs[16][16];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    long long sum = 0;

    for (int t = 0; t < (K + 15) / 16; t++) {
        if (row < M && (t * 16 + tx) < K) {
            As[ty][tx] = A[row * K + (t * 16 + tx)] % MOD;
        } else {
            As[ty][tx] = 0;
        }

        if (col < N && (t * 16 + ty) < K) {
            Bs[ty][tx] = B[(t * 16 + ty) * N + col] % MOD;
        } else {
            Bs[ty][tx] = 0;
        }

        __syncthreads();

        for (int k_s = 0; k_s < 16; k_s++) {
            long long product = ((long long)As[ty][k_s] * Bs[k_s][tx]) % MOD;
            sum = (sum + product) % MOD;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = (long)(sum % MOD);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: This implementation requires at least 2 GPUs\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm cart_comm;
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int my_coords[2];
    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, my_coords);

    int p_rows = dims[0];
    int p_cols = dims[1];

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, my_coords[0], my_coords[1], &row_comm);
    MPI_Comm_split(cart_comm, my_coords[1], my_coords[0], &col_comm);

    unsigned long base_size = 32768;
    long multiplier = 1;
    
    unsigned long M = base_size * multiplier;
    unsigned long N = base_size * multiplier;
    unsigned long K = base_size * multiplier;

    if (cart_rank == 0) {
        printf("SUMMA Matrix Multiplication\n");
        printf("Matrix dimensions: %lu × %lu × %lu\n", M, N, K);
        printf("Process grid: %d × %d\n", p_rows, p_cols);
        printf("GPUs: %d\n", world_size);
    }

    if (M % p_rows != 0 || N % p_cols != 0) {
        if (cart_rank == 0) {
            fprintf(stderr, "Error: Matrix dimensions must be divisible by grid\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    long local_M = M / p_rows;
    long local_N = N / p_cols;
    long local_K = K / (p_rows == p_cols ? p_rows : (p_rows > p_cols ? p_cols : p_rows));

    int gpu_count;
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count));
    int my_gpu = cart_rank % gpu_count;
    CUDA_CHECK(cudaSetDevice(my_gpu));
    
    // Report initial GPU memory (before allocation)
    size_t free_mem_before, total_mem;
    cudaMemGetInfo(&free_mem_before, &total_mem);
    
    // Allocate host memory
    long* h_A_local = (long*)malloc(local_M * local_K * sizeof(long));
    long* h_B_local = (long*)malloc(local_K * local_N * sizeof(long));
    long* h_C_local = (long*)calloc(local_M * local_N, sizeof(long));
    
    if (!h_A_local || !h_B_local || !h_C_local) {
        fprintf(stderr, "Rank %d: Host memory allocation failed\n", cart_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize data
    if (cart_rank == 0) {
        printf("Initializing matrix blocks...\n");
    }
    
    for(long i = 0; i < local_M * local_K; ++i) {
        h_A_local[i] = 1 % MOD;
    }
    for(long i = 0; i < local_K * local_N; ++i) {
        h_B_local[i] = 1 % MOD;
    }
    
    // Allocate device memory
    long *d_A_local, *d_B_local, *d_C_local;
    long *d_A_bcast, *d_B_bcast, *d_C_temp;
    
    CUDA_CHECK(cudaMalloc(&d_A_local, local_M * local_K * sizeof(long)));
    CUDA_CHECK(cudaMalloc(&d_B_local, local_K * local_N * sizeof(long)));
    CUDA_CHECK(cudaMalloc(&d_C_local, local_M * local_N * sizeof(long)));
    CUDA_CHECK(cudaMemset(d_C_local, 0, local_M * local_N * sizeof(long)));

    CUDA_CHECK(cudaMalloc(&d_A_bcast, local_M * local_K * sizeof(long)));
    CUDA_CHECK(cudaMalloc(&d_B_bcast, local_K * local_N * sizeof(long)));
    CUDA_CHECK(cudaMalloc(&d_C_temp, local_M * local_N * sizeof(long)));

    // Allocate broadcast buffers
    long* h_A_bcast = (long*)malloc(local_M * local_K * sizeof(long));
    long* h_B_bcast = (long*)malloc(local_K * local_N * sizeof(long));
    
    // Report GPU memory after allocation
    size_t free_mem_after, total_mem_after;
    cudaMemGetInfo(&free_mem_after, &total_mem_after);
    
    // Calculate actual memory used
    double mem_used_gb = (free_mem_before - free_mem_after) / (1024.0*1024.0*1024.0);
    printf("Rank %d: GPU %d - Memory used: %.2f GB, Free: %.2f GB / %.2f GB total\n",
           cart_rank, my_gpu, mem_used_gb,
           free_mem_after/(1024.0*1024.0*1024.0), 
           total_mem/(1024.0*1024.0*1024.0));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A_local, h_A_local, local_M * local_K * sizeof(long), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_local, h_B_local, local_K * local_N * sizeof(long), 
                          cudaMemcpyHostToDevice));
    
    MPI_Barrier(cart_comm);
    double start_time = MPI_Wtime();

    // SUMMA main loop
    dim3 block_dim(16, 16);
    dim3 grid_dim((local_N + 15) / 16, (local_M + 15) / 16);
    
    int num_k_iterations;
    if (p_rows == p_cols) {
        num_k_iterations = p_rows;
    } else if (world_size == 2) {
        num_k_iterations = 1;
    } else if (world_size == 8 && (p_rows == 2 && p_cols == 4)) {
        num_k_iterations = 2;
    } else if (world_size == 8 && (p_rows == 4 && p_cols == 2)) {
        num_k_iterations = 2;
    } else {
        num_k_iterations = (p_rows < p_cols) ? p_rows : p_cols;
    }
    
    for (int k_iter = 0; k_iter < num_k_iterations; ++k_iter) {
        int root_A = k_iter % p_cols;
        if (my_coords[1] == root_A) {
            CUDA_CHECK(cudaMemcpy(h_A_bcast, d_A_local, 
                                  local_M * local_K * sizeof(long), 
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(h_A_bcast, local_M * local_K, MPI_LONG, root_A, row_comm);
        CUDA_CHECK(cudaMemcpy(d_A_bcast, h_A_bcast, 
                              local_M * local_K * sizeof(long), 
                              cudaMemcpyHostToDevice));

        int root_B = k_iter % p_rows;
        if (my_coords[0] == root_B) {
            CUDA_CHECK(cudaMemcpy(h_B_bcast, d_B_local, 
                                  local_K * local_N * sizeof(long), 
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(h_B_bcast, local_K * local_N, MPI_LONG, root_B, col_comm);
        CUDA_CHECK(cudaMemcpy(d_B_bcast, h_B_bcast, 
                              local_K * local_N * sizeof(long), 
                              cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemset(d_C_temp, 0, local_M * local_N * sizeof(long)));
        
        matmul_kernel<<<grid_dim, block_dim>>>(
            d_A_bcast, d_B_bcast, d_C_temp, local_M, local_N, local_K);
        
        add_matrices_kernel<<<(local_M * local_N + 255) / 256, 256>>>(
            d_C_local, d_C_temp, local_M * local_N);
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(cart_comm);
    double end_time = MPI_Wtime();

    CUDA_CHECK(cudaMemcpy(h_C_local, d_C_local, 
                          local_M * local_N * sizeof(long), 
                          cudaMemcpyDeviceToHost));
    
    long expected_value = (K % MOD);
    int local_errors = 0;
    for (int i = 0; i < (local_M * local_N < 100 ? local_M * local_N : 100); ++i) {
        if (h_C_local[i] != expected_value) {
            local_errors++;
        }
    }
    
    int total_errors;
    MPI_Reduce(&local_errors, &total_errors, 1, MPI_INT, MPI_SUM, 0, cart_comm);
    
    if (cart_rank == 0) {
        double time_elapsed = end_time - start_time;
        double gflops = (2.0 * M * N * K) / (time_elapsed * 1e9);
        
        printf("\n======= Results =======\n");
        printf("Time: %.4f seconds\n", time_elapsed);
        printf("Performance: %.2f GFLOPS\n", gflops);
        
        if (total_errors == 0) {
            printf("Verification PASSED (expected: %ld)\n", expected_value);
        } else {
            printf("Verification FAILED (%d errors)\n", total_errors);
        }
    }

    free(h_A_local);
    free(h_B_local);
    free(h_C_local);
    free(h_A_bcast);
    free(h_B_bcast);

    cudaFree(d_A_local);
    cudaFree(d_B_local);
    cudaFree(d_C_local);
    cudaFree(d_C_temp);
    cudaFree(d_A_bcast);
    cudaFree(d_B_bcast);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}