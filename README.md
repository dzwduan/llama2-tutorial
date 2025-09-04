# llama2.c tutorial 


## Project: Background Knowledge

Re-implements neural network examples from Victor Zhou's blog using C. Includes single neuron and simple neural network implementations with comprehensive test cases.

See the README in the [Project: Background Knowledge](01-background-knowledge/README.md) folder.

## Project 2: llama2.c

Comprehensive unit testing achieving 100% line coverage for llama2.c, plus a complete Python reimplementation with generation and chat modes.

See the README in the [Project 2: llama2.c](02-llama2-test-and-rewrite/README.md) folder.

## Project 3: llama2.c with CUDA

Transforms llama2.c into a CUDA program with GPU matrix multiplication. Benchmarks against VLLM and CPU versions, achieving 15.7× speedup over CPU.

See the README in the [Project 3: llama2.c with CUDA](03-llama2c-cuda/README.md) folder.

## Project 4: CUDA Practice

CUDA implementations for 6 essential GPU operations: matrix multiplication, matrix transpose, ReLU activation, Leaky ReLU, reduction, and softmax. All test cases adapted from LeetGPU examples with Modal cloud testing.

See the README in the [Project 4: CUDA Practice](04-cuda-practice/README.md) folder.

## Project 5: Flash Attention

Implements three CUDA kernels for LLaMA2 inference: a softmax kernel, a FlashAttention V1 kernel, and FlashAttention + GEMV. Performance is tested against vLLM on A100 GPU, achieving 1.85× speedup with FlashAttention and 8.51× speedup with FlashAttention + GEMV over baseline matrix multiplication.

See the README in the [Project 5: Flash Attention](05-fast-attention/README.md) folder.


## REF

[INFO7375 Class]( https://github.com/dexkum-2myzZy-jipzid/INFO7375-HPC-AI.git)