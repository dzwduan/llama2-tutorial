# Project: CUDA Practice

This project provides CUDA implementations for 6 essential GPU operations with comprehensive testing on Modal cloud platform.

## Implementation Notes

- **Test Cases**: All test cases are adapted from LeetGPU examples
- **Testing Flow**: upload local files to cloud → compile CUDA files with nvcc → run tests on A10G GPU → report results → auto-cleanup container

## Files

- `solve.h`: Function declarations
- `modal_app.py`: Cloud testing platform
- `test_runner.cu`: Unified test runner
- `matrix_multiplication/*.cu`, `matrix_transpose/*.cu`, `relu_activation/*.cu`, `leaky_relu/*.cu`, `reduction/*.cu`, `softmax/*.cu`: CUDA implementation and test files

## Quick Start

### Change to Subdirectory

```bash
cd 04-cuda-practice
```

### Testing

1. **Run all operations (recommended):**

   ```bash
   modal run modal_app.py
   ```

2. **Run individual operations:**

   ```bash
   modal run modal_app.py --test-type matrix_mult
   modal run modal_app.py --test-type matrix_transpose
   modal run modal_app.py --test-type relu
   modal run modal_app.py --test-type leaky_relu
   modal run modal_app.py --test-type reduction
   modal run modal_app.py --test-type softmax
   ```
