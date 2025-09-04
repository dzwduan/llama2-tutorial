import json
import matplotlib.pyplot as plt
import numpy as np
import os

# A100‑SXM4‑40GB specs (Ampere GA100)
A100_PEAK_FP64 = 9.7 * 1e3  # FP64: 9.7 TFLOPS
A100_PEAK_FP32 = 19.5 * 1e3  # FP32: 19.5 TFLOPS
A100_PEAK_TF32 = (
    312 * 1e3
)  # TF32 Tensor Core: 312 TFLOPS (with sparsity, vLLM uses TF32)
A100_PEAK_FP16 = 624 * 1e3  # FP16/BF16 Tensor Core: 624 TFLOPS (with sparsity)
A100_PEAK_BANDWIDTH = 2039  # GB/s (H100 is ~3.35TB/s, A100 is ~2TB/s)
LLAMA7B_PARAMS = 7e9
LLAMA7B_FLOPS_PER_TOKEN = 2 * LLAMA7B_PARAMS


def plot_roofline(results_data):
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOPs/Byte)", fontsize=14)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=14)
    ax.set_title("A100 Roofline - Llama-2-7B Performance", fontsize=16)

    x = np.logspace(-1, 3, 100)
    ax.set_xlim(10**-1, 10**3)
    ax.set_ylim(10**1, 10**6)

    # Use TF32 peak if vLLM is present, otherwise use FP32
    peak_compute = A100_PEAK_FP32
    if any("vllm" in name for name in results_data):
        peak_compute = A100_PEAK_TF32  # Use TF32 peak if vLLM is included

    ridge_point = peak_compute / A100_PEAK_BANDWIDTH
    y_compute = np.full_like(x, peak_compute)
    y_memory = x * A100_PEAK_BANDWIDTH
    roof = np.minimum(y_compute, y_memory)

    ax.plot(x, roof, "k-", linewidth=3, label="A100 Peak")
    ax.text(
        ridge_point * 1.5,
        peak_compute,
        f"Compute Bound\n{peak_compute/1e3:.0f} TFLOP/s",
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.text(
        0.15,
        0.15 * A100_PEAK_BANDWIDTH,
        f"Memory Bound\n{A100_PEAK_BANDWIDTH} GB/s",
        rotation=45,
        rotation_mode="anchor",
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Implementation configs
    implementations = {
        "llama2-matmul.cu": {
            "color": "red",
            "marker": "o",
            "label": "matmul (baseline)",
        },
        "llama2-matmul-softmax.cu": {
            "color": "orange",
            "marker": "s",
            "label": "matmul + softmax",
        },
        "llama2-flashattention.cu": {
            "color": "blue",
            "marker": "^",
            "label": "flash attention",
        },
        "llama2-flashattention-gemv.cu": {
            "color": "purple",
            "marker": "*",
            "label": "flash attention + gemv",
        },
        "vllm": {"color": "green", "marker": "D", "label": "vLLM"},
    }

    # Same OI for all implementations
    bytes_per_token = LLAMA7B_PARAMS * 2  # bfloat16
    operational_intensity = LLAMA7B_FLOPS_PER_TOKEN / bytes_per_token

    # Plot points
    plotted_points = []  # Track positions to avoid overlap

    for impl_name, result in results_data.items():
        if result.get("success") and impl_name in implementations:
            tokens_per_second = result["tokens_per_second"]
            time_per_token = 1.0 / tokens_per_second
            achieved_gflops = (LLAMA7B_FLOPS_PER_TOKEN / time_per_token) / 1e9

            impl_config = implementations[impl_name]

            # Check for overlapping points and adjust position
            plot_x = operational_intensity
            plot_y = achieved_gflops

            # Add small offset if points are too close
            for prev_x, prev_y in plotted_points:
                if (
                    abs(np.log10(plot_x) - np.log10(prev_x)) < 0.1
                    and abs(np.log10(plot_y) - np.log10(prev_y)) < 0.1
                ):
                    plot_x *= 1.2  # Small horizontal offset
                    break

            plotted_points.append((plot_x, plot_y))

            ax.plot(
                plot_x,
                plot_y,
                impl_config["marker"],
                color=impl_config["color"],
                markersize=12,
                markeredgewidth=2,
                markeredgecolor="black",
                label=f'{impl_config["label"]} ({achieved_gflops:.1f} GFLOP/s)',
            )

            # Performance labels with better positioning
            ax.annotate(
                f"{tokens_per_second:.1f} tok/s",
                (plot_x, plot_y),
                xytext=(15, 15),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor=impl_config["color"], alpha=0.7
                ),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
            )

    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout()

    plt.savefig("roofline_chart.png", dpi=300, bbox_inches="tight")
    print("Chart saved to roofline_chart.png")


def main():
    if not os.path.exists("all_benchmark_results.json"):
        print("Error: all_benchmark_results.json not found")
        print("Run the benchmark script first")
        return

    with open("all_benchmark_results.json", "r") as f:
        data = json.load(f)

    print("A100 Roofline Analysis - Llama-2-7B")
    print("=" * 40)

    # Calc operational intensity (same for all)
    bytes_per_token = LLAMA7B_PARAMS * 2  # bfloat16
    operational_intensity = LLAMA7B_FLOPS_PER_TOKEN / bytes_per_token

    print(f"Model: {LLAMA7B_PARAMS/1e9:.1f}B params")
    print(f"FLOPs/token: {LLAMA7B_FLOPS_PER_TOKEN/1e9:.1f}G")
    print(f"Bytes/token: {bytes_per_token/1e9:.1f}GB")
    print(f"Op. intensity: {operational_intensity:.2f} FLOPs/Byte")
    print()

    successful_results = {}
    for impl_name, result in data.items():
        if result.get("success"):
            tokens_per_second = result["tokens_per_second"]
            time_per_token = 1.0 / tokens_per_second
            achieved_gflops = (LLAMA7B_FLOPS_PER_TOKEN / time_per_token) / 1e9

            successful_results[impl_name] = result

            # A100 efficiency
            efficiency = (
                achieved_gflops / A100_PEAK_FP32
            ) * 100  # compare to FP32 peak

            print(f"{impl_name}:")
            print(f"  {tokens_per_second:.1f} tok/s")
            print(f"  {achieved_gflops:.1f} GFLOP/s")
            print(f"  {efficiency:.2f}% A100 efficiency")

            # Speedup vs baseline
            if impl_name != "llama2-matmul.cu" and "llama2-matmul.cu" in data:
                baseline_tps = data["llama2-matmul.cu"]["tokens_per_second"]
                speedup = tokens_per_second / baseline_tps
                print(f"  {speedup:.2f}x speedup")
            print()

    if successful_results:
        plot_roofline(successful_results)

        # Summary
        summary_peak = (
            A100_PEAK_TF32 if "vllm" in successful_results else A100_PEAK_FP32
        )
        print(
            f"- A100 peak: {summary_peak/1e3:.1f} TFLOP/s, bandwidth: {A100_PEAK_BANDWIDTH} GB/s"
        )
        print(f"- Ridge point: {summary_peak/A100_PEAK_BANDWIDTH:.1f} FLOPs/Byte")
        print(
            f"- All implementations are memory bound (OI = {operational_intensity:.2f})"
        )
    else:
        print("No successful results found")


if __name__ == "__main__":
    main()
