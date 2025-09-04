import subprocess
import time
from pathlib import Path

import modal

app = modal.App("llama2c-cuda-benchmark")

# CUDA environment image
cuda_version = "12.2.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.11"
).apt_install("git", "build-essential", "wget")

# Create volume to store model files
volume = modal.Volume.from_name("llama2c-models", create_if_missing=True)


@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=3600,
    memory=32768,
    cpu=1,
    volumes={"/models": volume},
)
def benchmark_llama2c_cuda(run_cu_content: str):
    import os

    print("🚀 Starting llama2.c CUDA benchmark test...")

    # 1. Clone original llama2.c repository
    print("📥 Cloning llama2.c repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/karpathy/llama2.c.git", "/tmp/llama2.c"],
        check=True,
    )

    os.chdir("/tmp/llama2.c")

    # 2. Check CUDA environment
    print("🔍 Checking CUDA environment...")
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print("GPU Info:")
        print(result.stdout)
    except:
        print("⚠️  Warning: nvidia-smi command failed")

    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        print("NVCC Version:")
        print(result.stdout)
    except:
        return {
            "error": "NVCC not found - CUDA compilation environment not properly installed"
        }

    # 3. Write the modified run.cu content
    print("📝 Writing run.cu file...")
    try:
        with open("run.cu", "w") as f:
            f.write(run_cu_content)
        print("✅ Successfully written run.cu file")

    except Exception as e:
        return {
            "error": "Failed to write run.cu file",
            "message": str(e),
        }

    # 4. Check model files
    print("📦 Checking model and tokenizer files...")

    print("Files in /models directory:")
    try:
        for root, dirs, files in os.walk("/models"):
            for file in files:
                print(f"  {os.path.join(root, file)}")
    except:
        print("  Could not list directory contents")

    model_path = "/models/llama2-7b-chat.bin"
    tokenizer_path = "/models/tokenizer.bin"

    if not os.path.exists(model_path):
        return {
            "error": f"Model file not found at {model_path}",
            "suggestion": "Make sure the model file is uploaded to the volume",
        }

    print(f"✅ Model file confirmed: {model_path}")
    if os.path.exists(tokenizer_path):
        print(f"✅ Tokenizer file confirmed: {tokenizer_path}")
    else:
        print(f"⚠️  Tokenizer file not found at: {tokenizer_path}")

    # 5. Compile CUDA version
    print("🔨 Compiling CUDA version...")

    cuda_compile = subprocess.run(
        [
            "nvcc",
            "-O3",
            "-o",
            "runcu",
            "run.cu",
            "-lcudart",
            "-lm",
        ],
        capture_output=True,
        text=True,
    )

    if cuda_compile.returncode != 0:
        return {
            "error": "CUDA compilation failed",
            "compile_stdout": cuda_compile.stdout,
            "compile_stderr": cuda_compile.stderr,
            "suggestion": "Check CUDA syntax and header file includes in run.cu",
        }

    print("✅ CUDA version compiled successfully")

    # 6. Test basic program execution first
    print("🧪 Testing basic program execution...")
    try:
        # First test: just run the program without any arguments to see if it starts
        test_cmd = ["./runcu"]
        test_result = subprocess.run(
            test_cmd, capture_output=True, text=True, timeout=10
        )
        print(f"Basic test exit code: {test_result.returncode}")
        if test_result.stdout:
            print(f"Basic test stdout: {test_result.stdout[:200]}")
        if test_result.stderr:
            print(f"Basic test stderr: {test_result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("⚠️  Basic test timed out - program may be hanging")
        return {
            "error": "Program hangs on startup - possible infinite loop or GPU initialization issue",
            "suggestion": "Check your CUDA initialization code and main function",
        }
    except Exception as e:
        print(f"⚠️  Basic test failed: {e}")

    # 7. GPU warm-up run with shorter timeout
    print("🔥 GPU warm-up run...")
    try:
        warmup_cmd = ["./runcu", model_path, "-n", "5", "-t", "1.0", "-i", "Hi"]
        if os.path.exists(tokenizer_path):
            warmup_cmd.extend(["-z", tokenizer_path])

        print(f"Warmup command: {' '.join(warmup_cmd)}")
        warmup_result = subprocess.run(
            warmup_cmd, capture_output=True, text=True, timeout=30
        )
        print("✅ GPU warm-up completed")
        print(f"Warmup exit code: {warmup_result.returncode}")
        if warmup_result.stdout:
            print(f"Warmup stdout preview: {warmup_result.stdout[:100]}")
    except subprocess.TimeoutExpired:
        print("⚠️  GPU warm-up timed out after 30s")
        return {
            "error": "GPU warm-up timed out - program is too slow or hanging",
            "suggestion": "Check CUDA memory allocation and kernel execution in your code",
        }
    except Exception as e:
        print(f"⚠️  GPU warm-up failed: {e}")
        return {
            "error": f"GPU warm-up failed: {str(e)}",
            "suggestion": "Check program arguments and model file compatibility",
        }

    # 7. Run benchmark test (run 5 times)
    print("⚡ Starting CUDA performance benchmark...")

    num_tokens = 100
    temperature = 0.8
    prompt = "Once upon a time,"
    num_runs = 5  # num of benchmark runs
    seed = 42  # fixed seed

    print(f"🔄 Running {num_runs} benchmark iterations...")

    # Define extraction functions
    def extract_generated_text(output):
        if not output:
            return ""

        lines = output.split("\n")
        text_lines = []
        for line in lines:
            if (
                line.strip()
                and not line.startswith("<")
                and not line.startswith("achieved")
                and not line.startswith("tok/s")
                and not "malloc" in line.lower()
                and not line.startswith("inference")
                and not line.startswith("temperature")
                and not "CUDA" in line
            ):
                text_lines.append(line.strip())

        full_text = " ".join(text_lines)
        return full_text

    def extract_tokens_per_second(output):
        if not output:
            return 0.0

        lines = output.split("\n")
        for line in lines:
            if "achieved tok/s:" in line:
                try:
                    tokens_per_sec = float(line.split("achieved tok/s:")[-1].strip())
                    return tokens_per_sec
                except:
                    pass

        return (
            num_tokens / cuda_time if "cuda_time" in locals() and cuda_time > 0 else 0.0
        )

    benchmark_results = []
    all_outputs = []

    cuda_cmd = [
        "./runcu",
        model_path,
        "-n",
        str(num_tokens),
        "-t",
        str(temperature),
        "-s",
        str(seed),
        "-i",
        prompt,
    ]
    if os.path.exists(tokenizer_path):
        cuda_cmd.extend(["-z", tokenizer_path])

    print(f"Command: {' '.join(cuda_cmd)}")

    for run_idx in range(num_runs):
        print(f"🚀 Running CUDA inference #{run_idx + 1}/{num_runs}...")
        start_time = time.time()

        cuda_result = subprocess.run(
            cuda_cmd, capture_output=True, text=True, timeout=600
        )

        cuda_time = time.time() - start_time
        cuda_success = cuda_result.returncode == 0

        if not cuda_success:
            print(f"❌ Run #{run_idx + 1} failed!")
            benchmark_results.append(
                {
                    "run_number": run_idx + 1,
                    "success": False,
                    "time_seconds": cuda_time,
                    "tokens_per_second": 0.0,
                    "error": f"Exit code: {cuda_result.returncode}",
                }
            )
            continue

        # Extract results for this run
        cuda_output = extract_generated_text(cuda_result.stdout)
        cuda_tokens_per_sec = extract_tokens_per_second(cuda_result.stderr)
        if cuda_tokens_per_sec == 0.0:
            cuda_tokens_per_sec = extract_tokens_per_second(cuda_result.stdout)

        benchmark_results.append(
            {
                "run_number": run_idx + 1,
                "success": True,
                "time_seconds": cuda_time,
                "tokens_per_second": cuda_tokens_per_sec,
                "output_length_chars": len(cuda_output),
                "output_length_words": len(cuda_output.split()) if cuda_output else 0,
            }
        )

        all_outputs.append(cuda_output)
        print(
            f"✅ Run #{run_idx + 1} completed: {cuda_time:.3f}s, {cuda_tokens_per_sec:.1f} tok/s"
        )

    # Calculate statistics
    successful_runs = [r for r in benchmark_results if r["success"]]

    if not successful_runs:
        return {
            "error": "All benchmark runs failed",
            "debug_info": {
                "cuda_stderr": cuda_result.stderr if "cuda_result" in locals() else "",
                "cuda_stdout": cuda_result.stdout if "cuda_result" in locals() else "",
                "failed_runs": benchmark_results,
            },
        }

    times = [r["time_seconds"] for r in successful_runs]
    tokens_per_sec = [r["tokens_per_second"] for r in successful_runs]

    import statistics

    time_stats = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std_dev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }

    tokens_per_sec_stats = {
        "mean": statistics.mean(tokens_per_sec),
        "median": statistics.median(tokens_per_sec),
        "min": min(tokens_per_sec),
        "max": max(tokens_per_sec),
        "std_dev": statistics.stdev(tokens_per_sec) if len(tokens_per_sec) > 1 else 0.0,
    }

    # 10. Organize results
    cuda_result_formatted = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "device": "A100 GPU",
        "framework": "llama2.c-CUDA",
        "prompt": prompt,
        "sample_output": all_outputs[0] if all_outputs else "",
        "benchmark_config": {
            "num_runs": num_runs,
            "successful_runs": len(successful_runs),
            "failed_runs": num_runs - len(successful_runs),
            "estimated_output_tokens": num_tokens,
            "temperature": temperature,
            "seed": seed,
        },
        "performance_stats": {
            "time_seconds": time_stats,
            "tokens_per_second": tokens_per_sec_stats,
        },
        "individual_runs": benchmark_results,
        "compilation_success": True,
        "execution_success": len(successful_runs) > 0,
    }

    if all_outputs:
        cuda_result_formatted["output_length_chars"] = len(all_outputs[0])
        cuda_result_formatted["output_length_words"] = (
            len(all_outputs[0].split()) if all_outputs[0] else 0
        )

    return cuda_result_formatted


@app.local_entrypoint()
def main():
    import sys
    import json

    # Check if local run.cu file exists
    if not Path("run.cu").exists():
        print("❌ Error: run.cu file not found")
        print(
            "Please ensure your modified run.cu file is in the same directory as this Python script"
        )
        sys.exit(1)

    print("🚀 Starting llama2.c CUDA benchmark test...")
    print("📁 Local run.cu file detected")

    # Read the local run.cu file
    with open("run.cu", "r") as f:
        run_cu_content = f.read()

    print("🔄 Starting Modal remote execution...")

    # Run benchmark test
    result = benchmark_llama2c_cuda.remote(run_cu_content)

    # Check for errors
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        if "compile_stderr" in result:
            print(f"\n📋 Compilation error details:")
            print(result["compile_stderr"])
        if "suggestion" in result:
            print(f"\n💡 Suggestion: {result['suggestion']}")
        return

    # Display results
    print(f"\n🎉 Benchmark test completed!\n")

    print(f"📊 CUDA inference results:")
    print(f"   Model: {result['model']}")
    print(f"   Device: {result['device']}")
    print(f"   Framework: {result['framework']}")
    print(f"   Prompt: {result['prompt']}")
    print(f"   Status: {'✅ Success' if result['execution_success'] else '❌ Failed'}")

    if result["execution_success"]:
        config = result["benchmark_config"]
        time_stats = result["performance_stats"]["time_seconds"]
        tokens_stats = result["performance_stats"]["tokens_per_second"]

        print(
            f"\n⚡ Performance metrics ({config['successful_runs']}/{config['num_runs']} successful runs):"
        )
        print(f"   Time (seconds):")
        print(f"     Mean: {time_stats['mean']:.3f}s ± {time_stats['std_dev']:.3f}s")
        print(f"     Range: {time_stats['min']:.3f}s - {time_stats['max']:.3f}s")
        print(f"     Median: {time_stats['median']:.3f}s")
        print(f"   Inference speed (tokens/sec):")
        print(
            f"     Mean: {tokens_stats['mean']:.1f} ± {tokens_stats['std_dev']:.1f} tok/s"
        )
        print(
            f"     Range: {tokens_stats['min']:.1f} - {tokens_stats['max']:.1f} tok/s"
        )
        print(f"     Median: {tokens_stats['median']:.1f} tok/s")
        print(f"   Generated tokens: {config['estimated_output_tokens']}")

        if "output_length_chars" in result:
            print(
                f"   Sample output length: {result['output_length_chars']} chars, {result['output_length_words']} words"
            )

    print(f"\n📝 Sample generated content:")
    sample_output = result.get("sample_output", "")
    if len(sample_output) > 500:
        print(f"{sample_output[:500]}...\n[Content truncated]")
    else:
        print(sample_output)

    print(f"\n📋 Complete results (JSON format):")
    display_result = result.copy()
    if "debug_info" in display_result:
        display_result["debug_info"] = "<<Debug info omitted>>"
    print(json.dumps(display_result, indent=2, ensure_ascii=False))

    # Save results to JSON file
    output_filename = f"benchmark_result_{int(time.time())}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_filename}")

    # If there is debug info
    if "debug_info" in result and not result["execution_success"]:
        print(f"\n⚠️  Debug information:")
        debug_info = result["debug_info"]
        if debug_info.get("cuda_stderr"):
            print(f"\n[CUDA error output]:")
            print(debug_info["cuda_stderr"])
        if debug_info.get("cuda_stdout"):
            print(f"\n[CUDA standard output]:")
            print(debug_info["cuda_stdout"])
        print(f"\n[Exit code]: {debug_info.get('cuda_exit_code', 'N/A')}")

    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    # Run the benchmark
    with app.run():
        main()
