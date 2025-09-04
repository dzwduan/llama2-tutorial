# Project: Llama2 Test and Rewrite

This project provides comprehensive unit tests for llama2.c achieving 100% line coverage, and a complete Python reimplementation.

## Files

### C Unit Tests

- `test_llama2.c`: Unit test file for run.c
- `out/`: Coverage report folder
- `coverage.info`: LCOV coverage data
- `Makefile`: Build and test automation

### Python Reimplementation  

- `run.py`: Python reimplementation of run.c

## Quick Start

### Change to Subdirectory

```bash
cd 02-llama2-test-and-rewrite
```

### Unit Tests

1. **Run tests and generate coverage report: (Optional)**

   ```bash
   make test
   make coverage
   ```

2. **View coverage report:**

   ```bash
   open out/index.html
   ```

### Python Implementation

1. **Setup:**

   ```bash
   wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
   ```

2. **Run generation:**

   ```bash
   python run.py stories15M.bin -i "Once upon a time" -t 0.8 -n 100
   ```

   > `-i`: Input prompt for generation  `-t`: Temperature  `-n`: Max tokens to generate

3. **Run chat mode:**

   ```bash
   python run.py stories15M.bin -m chat -s "You are a helpful assistant"
   ```

   > `-m`: Mode (generate/chat)   `-s`: System prompt for chat
