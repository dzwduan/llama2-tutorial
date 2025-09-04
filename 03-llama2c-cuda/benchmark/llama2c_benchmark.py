import modal
import time
import json
import subprocess

llama2c_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .run_commands(
        "git clone https://github.com/karpathy/llama2.c", "cd llama2.c && make runfast"
    )
)

app = modal.App("llama2c-benchmark")

model_volume = modal.Volume.from_name("llama2c-models")

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


@app.function(
    image=llama2c_image,
    gpu=None,
    cpu=1,
    memory=32768,
    timeout=30 * 60,
    volumes={
        "/models": model_volume,
    },
)
def run_existing_model():
    import os

    print("Checking for existing model files...")

    # Check if the model file already exists
    # model conversion is done separately to save time
    model_path = "/models/llama2-7b-chat.bin"
    tokenizer_path = "/models/tokenizer.bin"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Available files:", os.listdir("/models"))
        return None

    print(f"Found model: {os.path.getsize(model_path) / (1024**3):.2f} GB")

    # Change to llama2.c directory
    os.chdir("/llama2.c")
    print("Changed to llama2.c directory")

    # Benchmark configuration
    prompt = "Once upon a time,"
    num_tokens = 100
    temperature = 0.8
    seed = 42  # fix seed

    print("\nRunning llama2.c benchmark...")
    print(f"Prompt: '{prompt}'")
    print(f"Tokens to generate: {num_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")

    cmd = [
        "./run",
        model_path,
        "-i",
        prompt,
        "-n",
        str(num_tokens),
        "-t",
        str(temperature),
        "-s",
        str(seed),  # fixed seed
    ]

    # Add tokenizer path if exists
    if os.path.exists(tokenizer_path):
        cmd.extend(["-z", tokenizer_path])

    print(f"Command: {' '.join(cmd)}")

    # Run benchmark
    start_time = time.time()

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    # Collect output
    output_lines = []
    print("\nGenerating text:")
    print("-" * 50)

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            line = line.rstrip()
            output_lines.append(line)
            print(line)

    stderr = process.stderr.read()
    process.wait()

    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 50)
    print(f"\nSTDERR: {stderr}")

    output_text = "\n".join(output_lines)

    tokens_per_second = 0.0
    if "achieved tok/s:" in stderr:
        try:
            tok_s_part = stderr.split("achieved tok/s:")[1].strip()
            tokens_per_second = float(tok_s_part.split()[0])
        except:
            pass

    # Extract generated text
    generated_text = output_text
    if prompt in generated_text:
        prompt_index = generated_text.find(prompt)
        generated_text = generated_text[prompt_index + len(prompt) :].strip()

    # Count tokens (approximation)
    num_tokens_generated = len(generated_text.split())

    # Results
    results = {
        "model": MODEL_NAME,
        "device": "CPU (1 core)",
        "framework": "llama2.c",
        "prompt": prompt,
        "output": generated_text,
        "total_time_seconds": total_time,
        "estimated_output_tokens": num_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "output_length_chars": len(generated_text),
        "output_length_words": len(generated_text.split()),
        "temperature": temperature,
        "seed": seed,
        "max_tokens": num_tokens,
    }

    print(f"\n=== llama2.c Benchmark Results ===")
    print(f"Model: {results['model']}")
    print(f"Device: {results['device']}")
    print(f"Temperature: {results['temperature']}")
    print(f"Seed: {results['seed']}")
    print(f"Max tokens: {results['max_tokens']}")
    print(f"Total time: {results['total_time_seconds']:.3f} seconds")
    print(f"Actual tokens generated: {results['estimated_output_tokens']}")
    print(f"Tokens per second: {results['tokens_per_second']:.2f}")
    print(f"Note: With fixed seed {seed}, this output should be reproducible")

    return results


@app.local_entrypoint()
def main():
    print("Running llama2.c with existing model on Modal...")
    print("Configuration: seed=42, temperature=0.8, max_tokens=100")

    results = run_existing_model.remote()

    if results:
        # Save results to file
        with open("llama2c_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to llama2c_benchmark_results.json")
        print(f"llama2.c Performance: {results['tokens_per_second']:.2f} tokens/second")
    else:
        print("Failed to run benchmark")


if __name__ == "__main__":
    main()
