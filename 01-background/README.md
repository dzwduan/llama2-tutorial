# Project: Background Knowledge

This project re-implements neural network examples from [Victor Zhou’s blog](https://victorzhou.com/blog/intro-to-neural-networks/) using C. Both the single neuron and simple neural network are included, with test cases. All code is tested.


## Files

- `neuron.c` / `neuron.h`: Single neuron
- `neural_network.c` / `neural_network.h`: Small neural network
- `test_neuron.c`, `test_neural_network.c`: Test cases
- `Makefile`: For build and run

## Quick Start

1. **Build everything:**

   ```sh
   make
   ```

2. **Run tests with make:**

   ```sh
   make run_test_neuron
   make run_test_neural_network
   ```

3. **Clean build files:**

   ```sh
   make clean
   ```
