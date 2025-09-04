# Multi-GPU matmul benchmark

import subprocess
from pathlib import Path
import json
import re
import modal

CODE_FILE_TO_TEST = "multi_gpu_matmul.cu"

BENCHMARK_CONFIGS = [
    (2, 2),
    (4, 4),
    (8, 8),
]

APP_DIR = "/app"
app = modal.App("multi-gpu-matmul-benchmark")
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("openmpi-bin", "libopenmpi-dev")
    .env(
        {
            "NCCL_DEBUG": "INFO",
            "OMPI_MCA_btl_vader_single_copy_mechanism": "none",
        }
    )
    .add_local_file(CODE_FILE_TO_TEST, remote_path=f"{APP_DIR}/{CODE_FILE_TO_TEST}")
)


def extract_performance_metrics(output):
    # Extract timing and GFLOPS
    time_match = re.search(r"Time:\s*([0-9.]+)\s*seconds", output)
    gflops_match = re.search(r"Performance:\s*([0-9.]+)\s*GFLOPS", output)

    # Extract grid dimensions
    grid_match = re.search(r"Process grid:\s*(\d+)\s*×\s*(\d+)", output)
    if not grid_match:
        grid_match = re.search(r"Process grid:\s*(\d+)\s*[xX×]\s*(\d+)", output)

    # Extract memory usage
    memory_used_pattern = r"Memory used:\s*([0-9.]+)\s*GB"
    memory_matches = re.findall(memory_used_pattern, output)

    # Calculate averages
    avg_memory_used = 0.0
    max_memory_used = 0.0
    if memory_matches:
        memory_values = [float(m) for m in memory_matches]
        avg_memory_used = sum(memory_values) / len(memory_values)
        max_memory_used = max(memory_values)

    return {
        "time_seconds": float(time_match.group(1)) if time_match else 0.0,
        "gflops": float(gflops_match.group(1)) if gflops_match else 0.0,
        "grid_dims": (
            (int(grid_match.group(1)), int(grid_match.group(2)))
            if grid_match
            else (0, 0)
        ),
        "avg_memory_used_gb": avg_memory_used,
        "max_memory_used_gb": max_memory_used,
        "num_gpus_reported": len(memory_matches),
    }


def do_compile_and_run(gpu_count: int, mpi_processes: int):
    import os

    code_path = os.path.join(APP_DIR, CODE_FILE_TO_TEST)
    output_binary = "multi_gpu_matmul.bin"

    compile_cmd = [
        "nvcc",
        "-arch=sm_80",
        "-O3",
        "-rdc=true",
        "-ccbin",
        "mpicc",
        "-o",
        output_binary,
        code_path,
        "-lm",
        "-lstdc++",
    ]
    print(f"Compiling: {' '.join(compile_cmd)}")
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)

    if compile_result.returncode != 0:
        return {
            "error": "Compilation failed",
            "gpu_count": gpu_count,
            "mpi_processes": mpi_processes,
            "compile_stderr": compile_result.stderr,
        }
    print("Compilation successful.")

    run_cmd = [
        "mpirun",
        "--allow-run-as-root",
        "-np",
        str(mpi_processes),
        f"./{output_binary}",
    ]
    print(f"Executing: {' '.join(run_cmd)}")
    run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=600)

    if run_result.returncode != 0:
        return {
            "error": "Execution failed",
            "gpu_count": gpu_count,
            "mpi_processes": mpi_processes,
        }
    print("Execution successful.")

    performance = extract_performance_metrics(run_result.stdout)
    return {
        "success": True,
        "gpu_count": gpu_count,
        "mpi_processes": mpi_processes,
        "time_seconds": performance["time_seconds"],
        "gflops": performance["gflops"],
        "grid_dims": performance["grid_dims"],
        "avg_memory_used_gb": performance["avg_memory_used_gb"],
        "max_memory_used_gb": performance["max_memory_used_gb"],
        "stdout": run_result.stdout,
    }


@app.function(image=cuda_image, gpu="A100-40GB:2", timeout=700)
def run_benchmark_2_gpu(mpi_processes: int):
    return do_compile_and_run(2, mpi_processes)


@app.function(image=cuda_image, gpu="A100-40GB:4", timeout=700)
def run_benchmark_4_gpu(mpi_processes: int):
    return do_compile_and_run(4, mpi_processes)


@app.function(image=cuda_image, gpu="A100-40GB:8", timeout=700)
def run_benchmark_8_gpu(mpi_processes: int):
    return do_compile_and_run(8, mpi_processes)


@app.local_entrypoint()
def main():
    if not Path(CODE_FILE_TO_TEST).exists():
        print(f"Error: Code file '{CODE_FILE_TO_TEST}' not found.")
        return

    print("=" * 80)
    print("Multi-GPU Matrix Multiplication Benchmark")
    print("=" * 80)
    print(f"Target file: {CODE_FILE_TO_TEST}")
    print(f"Configurations: {BENCHMARK_CONFIGS}")
    print("=" * 80)

    all_results = []

    benchmark_functions = {
        2: run_benchmark_2_gpu,
        4: run_benchmark_4_gpu,
        8: run_benchmark_8_gpu,
    }

    for gpu_count, mpi_processes in BENCHMARK_CONFIGS:
        print(f"\n--- Testing {gpu_count} GPU(s), {mpi_processes} MPI processes ---")

        target_function = benchmark_functions.get(gpu_count)
        if not target_function:
            print(f"SKIPPING: No function defined for {gpu_count} GPUs.")
            continue

        result = target_function.remote(mpi_processes)
        all_results.append(result)

    # Results table
    print("\n" + "=" * 110)
    print("Results Summary")
    print("=" * 110)
    print(
        f"{'GPUs':<6} | {'MPI':<6} | {'Grid':<8} | {'Time (s)':<10} | {'GFLOPS':<12} | "
        f"{'Speedup':<8} | {'Efficiency':<10} | {'Memory/GPU':<12}"
    )
    print("-" * 110)

    processed_results = [r for r in all_results]

    baseline_time = None
    baseline_gflops = None
    baseline_gpus = 2

    for result in processed_results:
        if result.get("success") and result["gpu_count"] == baseline_gpus:
            baseline_time = result["time_seconds"]
            baseline_gflops = result["gflops"]
            break

    for result in processed_results:
        if result.get("success"):
            gpu_str = str(result["gpu_count"])
            mpi_str = str(result["mpi_processes"])

            if result["grid_dims"][0] > 0:
                grid_str = f"{result['grid_dims'][0]}×{result['grid_dims'][1]}"
            else:
                if result["gpu_count"] == 2:
                    grid_str = "2×1"
                elif result["gpu_count"] == 4:
                    grid_str = "2×2"
                elif result["gpu_count"] == 8:
                    grid_str = "4×2"
                else:
                    grid_str = "Unknown"

            time_str = f"{result['time_seconds']:.4f}"
            gflops_str = f"{result['gflops']:.2f}"

            # Memory usage
            memory_str = (
                f"{result['avg_memory_used_gb']:.2f} GB"
                if result["avg_memory_used_gb"] > 0
                else "N/A"
            )

            if baseline_time and baseline_gflops:
                speedup = result["gflops"] / baseline_gflops
                relative_speedup = result["gflops"] / (
                    baseline_gflops * result["gpu_count"] / baseline_gpus
                )
                efficiency = relative_speedup * 100

                speedup_str = f"{speedup:.2f}x"
                efficiency_str = f"{efficiency:.1f}%"
            else:
                speedup_str = "N/A"
                efficiency_str = "N/A"

            print(
                f"{gpu_str:<6} | {mpi_str:<6} | {grid_str:<8} | {time_str:<10} | "
                f"{gflops_str:<12} | {speedup_str:<8} | {efficiency_str:<10} | "
                f"{memory_str:<12}"
            )
        else:
            error_msg = result.get("error", "Unknown error")
            gpu_str = str(result.get("gpu_count", "N/A"))
            mpi_str = str(result.get("mpi_processes", "N/A"))
            print(f"{gpu_str:<6} | {mpi_str:<6} | {'FAILED':<8} | {error_msg}")

    print("-" * 110)

    # Performance analysis
    print("\n" + "=" * 60)
    print("Performance Analysis")
    print("=" * 60)

    if baseline_gflops:
        print(f"Baseline (2 GPUs): {baseline_gflops:.2f} GFLOPS\n")
        for result in processed_results:
            if result.get("success"):
                actual_gflops = result["gflops"]
                expected_gflops = baseline_gflops * (
                    result["gpu_count"] / baseline_gpus
                )
                scaling_efficiency = (actual_gflops / expected_gflops) * 100

                print(
                    f"{result['gpu_count']} GPUs: {actual_gflops:.2f} GFLOPS "
                    f"(Scaling efficiency: {scaling_efficiency:.1f}%)"
                )

                if result["avg_memory_used_gb"] > 0:
                    print(f"  Memory per GPU: {result['avg_memory_used_gb']:.2f} GB")

    # Save report
    report_path = "multi_gpu_matmul_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to {report_path}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
