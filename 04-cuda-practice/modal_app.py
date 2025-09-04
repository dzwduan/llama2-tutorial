import modal
import os

app = modal.App("cuda-practice")

# Use official CUDA image and add python3 support
cuda_version = "12.3.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("python3-pip")
    .pip_install("numpy")
    # Copy source files
    .add_local_dir(".", "/app")
)


@app.function(
    gpu="A10G",
    image=cuda_image,
    timeout=300,
)
def run_single_test(test_type: str):
    """Run single test"""
    import subprocess
    import os

    os.chdir("/app")

    if test_type == "matrix_mult":
        print("Compiling matrix multiplication test...")
        compile_cmd = [
            "nvcc",
            "matrix_multiplication/test_matrix_multiplication.cu",
            "matrix_multiplication/matrix_multiplication.cu",
            "-o",
            "single_test",
            "-I.",
            "-DSTANDALONE_TEST",
        ]
        test_name = "Matrix Multiplication"

    elif test_type == "matrix_transpose":
        print("Compiling matrix transpose test...")
        compile_cmd = [
            "nvcc",
            "matrix_transpose/test_matrix_transpose.cu",
            "matrix_transpose/matrix_transpose.cu",
            "-o",
            "single_test",
            "-I.",
            "-DSTANDALONE_TEST",
        ]
        test_name = "Matrix Transpose"

    elif test_type == "relu":
        print("Compiling ReLU activation test...")
        compile_cmd = [
            "nvcc",
            "relu_activation/test_relu_activation.cu",
            "relu_activation/relu_activation.cu",
            "-o",
            "single_test",
            "-I.",
            "-DSTANDALONE_TEST",
        ]
        test_name = "ReLU Activation"

    elif test_type == "leaky_relu":
        print("Compiling Leaky ReLU test...")
        compile_cmd = [
            "nvcc",
            "leaky_relu/test_leaky_relu.cu",
            "leaky_relu/leaky_relu.cu",
            "-o",
            "single_test",
            "-I.",
            "-DSTANDALONE_TEST",
        ]
        test_name = "Leaky ReLU"

    elif test_type == "reduction":
        print("Compiling Reduction test...")
        compile_cmd = [
            "nvcc",
            "reduction/test_reduction.cu",
            "reduction/reduction.cu",
            "-o",
            "single_test",
            "-I.",
            "-DSTANDALONE_TEST",
        ]
        test_name = "Reduction"

    elif test_type == "softmax":
        print("Compiling Softmax test...")
        compile_cmd = [
            "nvcc",
            "softmax/test_softmax.cu",
            "softmax/softmax.cu",
            "-o",
            "single_test",
            "-I.",
            "-DSTANDALONE_TEST",
        ]
        test_name = "Softmax"
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Compilation successful")
    except subprocess.CalledProcessError as e:
        print("Compilation failed")
        print("Error:", e.stderr)
        raise

    print(f"\nRunning {test_name} test...")
    print("-" * 40)

    result = subprocess.run(["./single_test"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{test_name} test failed.")

    try:
        os.remove("single_test")
    except Exception as e:
        print(f"Warning: cleanup failed: {e}")

    print(f"{test_name} test completed")


@app.function(
    gpu="A10G",
    image=cuda_image,
    timeout=900,  # Increase timeout because there are many tests now
)
def run_all_tests():
    """Run comprehensive test with all operations"""
    import subprocess
    import os

    os.chdir("/app")

    print("Compiling all operations test...")
    compile_cmd = [
        "nvcc",
        "test_runner.cu",
        "matrix_multiplication/test_matrix_multiplication.cu",
        "matrix_transpose/test_matrix_transpose.cu",
        "relu_activation/test_relu_activation.cu",
        "leaky_relu/test_leaky_relu.cu",
        "reduction/test_reduction.cu",
        "softmax/test_softmax.cu",
        "matrix_multiplication/matrix_multiplication.cu",
        "matrix_transpose/matrix_transpose.cu",
        "relu_activation/relu_activation.cu",
        "leaky_relu/leaky_relu.cu",
        "reduction/reduction.cu",
        "softmax/softmax.cu",
        "-o",
        "all_operations_test",
        "-I.",
    ]

    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Compilation successful")
        print(
            f"Compiled {len([f for f in compile_cmd if f.endswith('.cu')])} CUDA files"
        )
    except subprocess.CalledProcessError as e:
        print("Compilation failed")
        print("Error:", e.stderr)
        print("Command:", " ".join(compile_cmd))
        raise

    print("\nRunning all CUDA operations test...")
    print("-" * 50)

    result = subprocess.run(["./all_operations_test"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Some tests failed.")

    try:
        os.remove("all_operations_test")
    except Exception as e:
        print(f"Warning: cleanup failed: {e}")

    print("🎉 All CUDA operations test completed!")


@app.local_entrypoint()
def main(test_type: str = "all"):
    """
    Run CUDA tests

    Args:
        test_type: Test type
            - "all": Run comprehensive test with all operations (default)
            - "matrix_mult": Matrix multiplication
            - "matrix_transpose": Matrix transpose
            - "relu": ReLU activation function
            - "leaky_relu": Leaky ReLU activation function
            - "reduction": Reduction sum
            - "softmax": Softmax function
    """

    print(f"Starting CUDA test: {test_type}")

    if test_type == "all":
        print("Running comprehensive test with all 6 operations...")
        run_all_tests.remote()

    elif test_type in [
        "matrix_mult",
        "matrix_transpose",
        "relu",
        "leaky_relu",
        "reduction",
        "softmax",
    ]:
        print(f"Running {test_type} test only...")
        run_single_test.remote(test_type)

    else:
        print(f"Unknown test type: {test_type}")
        print("Available options:")
        print("  - all: Run tests for all 6 operations")
        print("  - matrix_mult: Matrix multiplication")
        print("  - matrix_transpose: Matrix transpose")
        print("  - relu: ReLU activation function")
        print("  - leaky_relu: Leaky ReLU activation function")
        print("  - reduction: Reduction sum")
        print("  - softmax: Softmax function")
        return 1

    print("\nModal testing completed")
    return 0


if __name__ == "__main__":
    main()
