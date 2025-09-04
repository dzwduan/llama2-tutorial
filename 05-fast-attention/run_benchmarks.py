"""
Benchmark script for llama2.c CUDA implementations vs vLLM baseline.

Tests three CUDA implementations:
- llama2-matmul.cu (baseline)
- llama2-matmul-softmax.cu (with softmax optimization)
- llama2-flashattention.cu (with flash attention)
- llama2-flashattention-gemv.cu (fast attention with gemv)

All tests use the same parameters for fair comparison.

Usage:
    modal run run_benchmarks.py

"""

import subprocess
from pathlib import Path
import json
import shutil
import time

import modal

DEFAULT_FILES_TO_TEST = [
    "llama2-matmul.cu",
    "llama2-matmul-softmax.cu",
    "llama2-flashattention.cu",
    "llama2-flashattention-gemv.cu",
]

# vLLM config
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
INCLUDE_VLLM = True  # Set to True to include vLLM benchmark

# Test parameters
BENCHMARK_PROMPT = "Once upon a time,"
BENCHMARK_TOKENS = 100
BENCHMARK_SEED = 42
WARMUP_TOKENS = 10

APP_DIR = "/app"
app = modal.App("llama2c-benchmarks")

# CUDA setup
cuda_version = "12.2.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).apt_install("git", "build-essential", "wget")

# vLLM setup
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "transformers",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Storage volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volume = modal.Volume.from_name("llama2c-models", create_if_missing=True)

for filename in DEFAULT_FILES_TO_TEST:
    if filename.startswith("#"):
        continue
    cuda_image = cuda_image.add_local_file(
        filename, remote_path=f"{APP_DIR}/{filename}"
    )


@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=3600,
    volumes={"/models": volume},
)
def run_cuda_benchmark(version_name: str):
    import os

    print(f"benchmarking {version_name}")

    # Print basic GPU info only for the first benchmark
    if version_name == "llama2-matmul.cu":
        try:
            gpu_info = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if gpu_info.returncode == 0:
                gpu_data = gpu_info.stdout.strip().split(", ")
                print(f"GPU: {gpu_data[0]}")
                print(f"Memory: {gpu_data[1]} MB")
                print(f"Driver: {gpu_data[2]}")
            else:
                print("GPU info not available")
        except Exception:
            print("GPU info not available")

    compile_dir = "/tmp/llama2c"
    os.makedirs(compile_dir, exist_ok=True)
    os.chdir(compile_dir)
    if not os.path.exists(".git"):
        subprocess.run(
            ["git", "clone", "https://github.com/karpathy/llama2.c.git", "."],
            check=True,
        )

    source_cu_path = os.path.join(APP_DIR, version_name)
    target_cu_path = os.path.join(compile_dir, "run.cu")

    if not os.path.exists(source_cu_path):
        return {"error": f"File not found in container: {source_cu_path}"}

    shutil.copy(source_cu_path, target_cu_path)
    print(f"copied {version_name} -> run.cu")

    print("compiling...")
    compile_result = subprocess.run(
        ["nvcc", "-O3", "-o", "runcu", "run.cu", "-lcudart", "-lm"],
        capture_output=True,
        text=True,
    )

    if compile_result.returncode != 0:
        return {
            "error": "CUDA compilation failed",
            "version": version_name,
            "compile_stderr": compile_result.stderr,
        }
    print("compiled successfully")

    print("warming up...")
    model_path = "/models/llama2-7b-chat.bin"
    tokenizer_path = "/models/tokenizer.bin"

    warmup_cmd = ["./runcu", model_path, "-n", str(WARMUP_TOKENS), "-i", "warmup"]
    if os.path.exists(tokenizer_path):
        warmup_cmd.extend(["-z", tokenizer_path])

    subprocess.run(warmup_cmd, capture_output=True, text=True, timeout=120)
    print("warmup done")

    cuda_cmd = [
        "./runcu",
        model_path,
        "-n",
        str(BENCHMARK_TOKENS),
        "-i",
        BENCHMARK_PROMPT,
        "-s",
        str(BENCHMARK_SEED),
    ]
    if os.path.exists(tokenizer_path):
        cuda_cmd.extend(["-z", tokenizer_path])

    # run 3 times and average
    print(f"running {version_name} (3 runs)...")
    all_tokens_per_sec = []

    for run_num in range(3):
        print(f"  run {run_num + 1}/3...")
        run_result = subprocess.run(
            cuda_cmd, capture_output=True, text=True, timeout=600
        )

        if run_result.returncode != 0:
            return {
                "error": "Execution failed",
                "version": version_name,
                "exec_stderr": run_result.stderr,
            }

        def extract_tok_per_sec(output):
            for line in output.split("\n"):
                if "achieved tok/s:" in line:
                    try:
                        return float(line.split("achieved tok/s:")[-1].strip())
                    except (ValueError, IndexError):
                        continue
            return 0.0

        tokens_per_sec = extract_tok_per_sec(run_result.stderr) or extract_tok_per_sec(
            run_result.stdout
        )
        all_tokens_per_sec.append(tokens_per_sec)
        print(f"    {tokens_per_sec:.2f} tok/s")

    # get average
    avg_tokens_per_sec = sum(all_tokens_per_sec) / len(all_tokens_per_sec)
    print(
        f"{version_name}: {avg_tokens_per_sec:.2f} tok/s (avg of {all_tokens_per_sec})"
    )
    print(f"stdout:\n{run_result.stdout}")
    print(f"stderr:\n{run_result.stderr}")

    return {
        "version": version_name,
        "success": True,
        "tokens_per_second": avg_tokens_per_sec,
        "individual_runs": all_tokens_per_sec,
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }


@app.function(
    image=vllm_image,
    gpu="A100:1",
    timeout=10 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_vllm_benchmark():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=BENCHMARK_TOKENS, seed=BENCHMARK_SEED)
    prompt = BENCHMARK_PROMPT

    print("Warming up vLLM...")
    # warm up with same token count as CUDA
    llm.generate(
        [prompt], SamplingParams(max_tokens=WARMUP_TOKENS, seed=BENCHMARK_SEED)
    )

    print("Running vLLM benchmark (3 runs)...")
    all_tokens_per_sec = []

    for run_num in range(3):
        print(f"  vLLM run {run_num + 1}/3...")
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()

        total_time = end_time - start_time
        output_text = outputs[0].outputs[0].text
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        num_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

        if total_time > 0:
            tokens_per_second = num_tokens / total_time
            all_tokens_per_sec.append(tokens_per_second)
            print(f"    {tokens_per_second:.2f} tok/s")
        else:
            all_tokens_per_sec.append(0.0)
            print(f"    0.00 tok/s (failed)")

    # get average
    avg_tokens_per_sec = sum(all_tokens_per_sec) / len(all_tokens_per_sec)
    print(f"vLLM: {avg_tokens_per_sec:.2f} tok/s (avg of {all_tokens_per_sec})")

    return {
        "version": "vllm",
        "success": True,
        "tokens_per_second": avg_tokens_per_sec,
        "individual_runs": all_tokens_per_sec,
        "stdout": f"Generated {num_tokens} tokens using prompt: '{BENCHMARK_PROMPT}'",
        "stderr": f"Total time across {len(all_tokens_per_sec)} runs: {sum([t/(tps or 1) for t, tps in zip([num_tokens]*len(all_tokens_per_sec), all_tokens_per_sec)]):.2f}s",
    }


@app.local_entrypoint()
def main():
    files_to_test = DEFAULT_FILES_TO_TEST

    print("=" * 60)
    print("TEST CONFIG")
    print("=" * 60)
    print(f"Prompt: '{BENCHMARK_PROMPT}'")
    print(f"Tokens: {BENCHMARK_TOKENS}")
    print(f"Seed: {BENCHMARK_SEED}")
    print(f"Warmup: {WARMUP_TOKENS}")
    print("=" * 60)

    all_results = {}
    print(f"\nTesting CUDA implementations: {files_to_test}")

    # CUDA benchmarks
    for filename in files_to_test:
        print(f"\n--- {filename} ---")
        if not Path(filename).exists():
            print(f"skipping {filename} (not found)")
            all_results[filename] = {"error": "File not found"}
            continue

        result = run_cuda_benchmark.remote(filename)
        all_results[filename] = result

    # vLLM benchmark
    if INCLUDE_VLLM:
        print(f"\n--- vLLM Benchmark ---")
        try:
            vllm_result = run_vllm_benchmark.remote()
            all_results["vllm"] = vllm_result
        except Exception as e:
            print(f"vLLM benchmark failed: {e}")
            all_results["vllm"] = {"error": f"vLLM benchmark failed: {str(e)}"}

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Implementation':<30} | {'tok/s':<15}")
    print("-" * 60)

    successful_runs = []
    for version, result in all_results.items():
        if result.get("success"):
            perf_str = f"{result['tokens_per_second']:.2f}"
            successful_runs.append(result)
        else:
            error_msg = result.get("error", "Unknown error")
            perf_str = f"FAILED ({error_msg})"
        print(f"{version:<30} | {perf_str:<15}")
    print("-" * 60)

    # Speedup analysis
    if len(successful_runs) >= 2:
        cuda_results = [r for r in successful_runs if r["version"] != "vllm"]
        if cuda_results:
            baseline = cuda_results[0]
            print(f"\nSpeedup vs {baseline['version']}:")
            print("-" * 60)

            for result in successful_runs:
                if result["version"] != baseline["version"]:
                    if baseline["tokens_per_second"] > 0:
                        speedup = (
                            result["tokens_per_second"] / baseline["tokens_per_second"]
                        )
                        print(f"{result['version']:<30} | {speedup:.2f}x")
            print("-" * 60)

    with open("all_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to all_benchmark_results.json")
    print("Done.")

    # Summary
    print(f"\nSummary:")
    successful_cuda = [r for r in successful_runs if r["version"] != "vllm"]
    successful_vllm = [r for r in successful_runs if r["version"] == "vllm"]

    print(f"- CUDA tests: {len(successful_cuda)}/{len(files_to_test)} passed")
    if INCLUDE_VLLM:
        print(f"- vLLM test: {'passed' if successful_vllm else 'failed'}")
    print(f"- Total: {len(successful_runs)} successful runs")


if __name__ == "__main__":
    main()
