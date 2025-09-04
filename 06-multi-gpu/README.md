# Project 6: Multi-GPU Matrix Multiplication

This project implements a distributed matrix multiplication using the SUMMA (Scalable Universal Matrix Multiply Algorithm) algorithm across multiple GPUs with MPI and CUDA. The implementation can handle matrices too large for a single GPU by distributing computation across 2, 4, or 8 GPUs.

## Files

- `multi_gpu_matmul.cu`: SUMMA algorithm implementation with MPI + CUDA
- `multi-gpu-matmul-benchmark.py`: Modal cloud benchmarking script for multi-GPU configurations
- `multi_gpu_matmul_report.json`: Benchmark results and performance metrics

## Quick Start

### Change to Subdirectory

```bash
cd 06-multi-gpu
```

### Benchmarking

1. Run all benchmarks on Modal:

   ```bash
   modal run multi-gpu-matmul-benchmark.py | tee output.txt
   ```

   This will test configurations with 2, 4, and 8 GPUs automatically.

## Algorithm Details

### SUMMA Implementation

- **Matrix Dimensions**: 32768 × 32768 × 32768
- **Data Type**: `long` (8 bytes) with modulo arithmetic
- **Process Grids**:
  - 2 GPUs: 2×1 grid
  - 4 GPUs: 2×2 grid
  - 8 GPUs: 4×2 grid

### Key Features

- Distributes both matrices A and B across GPUs (tiling both matrices)
- Uses MPI for inter-GPU communication
- Verifies correctness of distributed computation

## Performance Results

### Scaling Performance

Based on benchmark results on NVIDIA A100-40GB GPUs:

| Configuration | Time (s) | GFLOPS | Speedup | Scaling Efficiency | Memory/GPU |
|--------------|----------|---------|---------|-------------------|------------|
| **2 GPUs (2×1)** | 96.69 | 727.80 | 1.00× | 100.0% (baseline) | 32.00 GB |
| **4 GPUs (2×2)** | 48.83 | 1441.05 | 1.98× | 99.0% | 12.00 GB |
| **8 GPUs (4×2)** | 29.21 | 2408.76 | 3.31× | 82.7% | 8.00 GB |
