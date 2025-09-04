# Project 3: llama2.c with CUDA

This project implements a CUDA-accelerated version of llama2.c and compares its performance against VLLM and the original CPU-based llama2.c implementation using the Llama-2-7b-chat-hf model.

## Files

- `run.cu`: CUDA version of llama2.c with GPU-accelerated `matmul`
- `llama2c_cuda_benchmark.py`: Modal script to run CUDA implementation using llama-2-7b-chat-hf weights (**pre-converted to llama2.c's custom format, stored in Modal cloud volume *llama2c-models* directory**)

- ```
  benchmark/
  ```

  : Results and benchmark scripts folder

  - `llama2c_benchmark_results.json`: CPU llama2.c results
  - `llama2c-cuda_benchmark.json`: CUDA implementation results
  - `vllm_benchmark_results.json`: VLLM benchmark results
  - `llama2c_benchmark.py`: Script for CPU llama2.c benchmarking
  - `vllm_benchmark.py`: VLLM performance testing script

## Quick Start

### Change to Subdirectory

```bash
cd 03-llama2c-cuda
```

### Running Implementation

1. **Run CUDA implementation:**

   ```bash
   modal run llama2c_cuda_benchmark.py | tee output.txt
   ```

## Performance Results

### Summary

| Implementation          | Device       | Tokens/Second | Speedup vs CPU |
| ----------------------- | ------------ | ------------- | -------------- |
| **VLLM**                | A100 GPU     | **88.45**     | 331×           |
| **llama2.c-CUDA**       | A100 GPU     | **4.19**      | 15.7×          |
| **llama2.c (non-CUDA)** | CPU (1 core) | **0.27**      | 1×             |

## Implementation Notes

- **Model**: meta-llama/Llama-2-7b-chat-hf
- **Weight Conversion**: Used llama2.c's `export.py` script for safetensors to custom format conversion
- **Deterministic Output**: Fixed seed (42) ensures consistent output across runs
- **Test Platform**: Modal cloud platform
- **Optimization Focus**: GPU-accelerated `matmul`

## Benchmark Configuration

- **Prompt**: "Once upon a time,"
- **Temperature**: 0.8
- **Seed**: 42 (fixed for reproducible results)
- **Max Tokens**: ~100
- **CUDA Runs**: 5 iterations with statistical analysis
