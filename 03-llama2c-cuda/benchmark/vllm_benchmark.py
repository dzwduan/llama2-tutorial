# VLLM Performance Benchmark on A100 GPU
# This script measures VLLM's tokens/second performance for comparison with llama2.c

import modal
import time
import json

# Container image setup
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.7.2",
        "huggingface_hub[hf_transfer]==0.26.2",
        "transformers",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("vllm-benchmark")

# Cache volumes for model weights
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


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
    import os
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not found. Please check your Modal secret setup."
        )

    from huggingface_hub import login

    login(token=hf_token)

    print("Loading VLLM model...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=100,
        stop=None,
        seed=42,  # fixed seed
    )

    # Benchmark prompt
    prompt = "Once upon a time,"

    print("Starting VLLM inference benchmark...")

    print("Warming up...")
    warmup_params = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=10,
        seed=42,
    )
    _ = llm.generate([prompt], warmup_params)

    # Actual benchmark run
    print("Running benchmark...")
    start_time = time.time()

    outputs = llm.generate([prompt], sampling_params)

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    output_text = outputs[0].outputs[0].text

    # Count tokens accurately using the model's tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    # Count actual tokens in the output
    output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
    num_tokens = len(output_tokens)
    tokens_per_second = num_tokens / total_time

    # Results
    results = {
        "model": MODEL_NAME,
        "device": "A100 GPU",
        "framework": "VLLM",
        "prompt": prompt,
        "output": output_text,
        "total_time_seconds": total_time,
        "estimated_output_tokens": num_tokens,
        "tokens_per_second": tokens_per_second,
        "output_length_chars": len(output_text),
        "output_length_words": len(output_text.split()),
        "seed": 42,
        "temperature": 0.8,
        "top_p": 0.9,
    }

    print(f"\n=== VLLM Benchmark Results ===")
    print(f"Model: {results['model']}")
    print(f"Device: {results['device']}")
    print(f"Seed: {results['seed']}")
    print(f"Total time: {results['total_time_seconds']:.3f} seconds")
    print(f"Actual tokens generated: {results['estimated_output_tokens']}")
    print(f"Tokens per second: {results['tokens_per_second']:.2f}")
    print(f"Output: {results['output'][:100]}...")

    return results


@app.local_entrypoint()
def test():
    print("Starting VLLM benchmark on A100 GPU...")

    results = run_vllm_benchmark.remote()

    # Save results to file
    with open("vllm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to vllm_benchmark_results.json")
    print(f"VLLM Performance: {results['tokens_per_second']:.2f} tokens/second")

    return results


if __name__ == "__main__":
    test()
