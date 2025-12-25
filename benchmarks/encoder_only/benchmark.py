#!/usr/bin/env python3
"""Benchmark script for image classification models on CPU.

Compares 4 modes:
- baseline: no optimizations
- autocast: torch.autocast only
- compile: torch.compile only
- autocast+compile: both optimizations
"""

import gc
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import timm
import torch
from tabulate import tabulate
from tqdm import tqdm

# Models to benchmark (reduced set for faster comparison)
MODELS = [
    "mobileone_s0",
    "mobileone_s1",
    "mobilenetv4_conv_small.e2400_r224_in1k",
    "mobilenetv4_conv_medium.e500_r224_in1k",
    "regnetx_002",
    "regnetx_004",
    "regnetx_008",
    "regnety_002",
    "regnety_004",
    "lcnet_100",
    "lcnet_150",
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    "resnet18",
    "resnet34",
    "repvgg_a0",
    "repvgg_a1",
    "efficientnet_b0",
    "ghostnetv2_100",
    "tinynet_b",
    "tinynet_c",
    "edgenext_x_small",
    "edgenext_small",
]

MODES = ["baseline", "autocast", "compile", "autocast+compile"]


@dataclass
class BenchmarkResult:
    model_name: str
    mode: str
    params_millions: float
    throughput_img_per_sec: float
    memory_mb_per_image: float


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def create_model(model_name: str, use_compile: bool = False):
    """Create and prepare model."""
    is_vit = "vit" in model_name or "deit" in model_name
    if is_vit:
        model = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
    else:
        model = timm.create_model(model_name, pretrained=False)
    model.eval()

    # Reparameterize if applicable
    if any(x in model_name for x in ["repvgg", "mobileone", "fastvit"]):
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()

    if use_compile:
        model = torch.compile(model)

    return model


def benchmark_model(
    model_name: str,
    mode: str,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> BenchmarkResult:
    """Benchmark a single model in a specific mode."""

    input_size = 512
    use_autocast = "autocast" in mode
    use_compile = "compile" in mode

    model = create_model(model_name, use_compile=use_compile)
    num_params = count_parameters(model)
    params_millions = num_params / 1e6

    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # Warmup
    if use_autocast:
        with torch.inference_mode(), torch.autocast(device_type="cpu"):
            for _ in range(num_warmup):
                _ = model(dummy_input)
    else:
        with torch.inference_mode():
            for _ in range(num_warmup):
                _ = model(dummy_input)

    # Measure memory
    model_memory_mb = (num_params * 4) / (1024 * 1024)
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    peak_mem = mem_before
    if use_autocast:
        with torch.inference_mode(), torch.autocast(device_type="cpu"):
            for _ in range(3):
                _ = model(dummy_input)
                current_mem = process.memory_info().rss / (1024 * 1024)
                peak_mem = max(peak_mem, current_mem)
    else:
        with torch.inference_mode():
            for _ in range(3):
                _ = model(dummy_input)
                current_mem = process.memory_info().rss / (1024 * 1024)
                peak_mem = max(peak_mem, current_mem)

    activation_memory = max(0, peak_mem - mem_before)
    memory_per_image = model_memory_mb + activation_memory / batch_size

    # Measure throughput
    gc.collect()
    times = []

    if use_autocast:
        with torch.inference_mode(), torch.autocast(device_type="cpu"):
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append(end - start)
    else:
        with torch.inference_mode():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append(end - start)

    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time

    del model
    del dummy_input
    gc.collect()

    return BenchmarkResult(
        model_name=model_name,
        mode=mode,
        params_millions=params_millions,
        throughput_img_per_sec=throughput,
        memory_mb_per_image=memory_per_image,
    )


def main():
    print("=" * 70)
    print("BENCHMARK ENCODERS CPU - Comparaison des modes d'optimisation")
    print("=" * 70)
    print(f"Modèles: {len(MODELS)}, Modes: {MODES}")
    print()

    results = []

    for model_name in tqdm(MODELS, desc="Models"):
        for mode in MODES:
            try:
                result = benchmark_model(model_name, mode)
                results.append(result)
            except Exception as e:
                tqdm.write(f"  ERREUR {model_name} [{mode}]: {e}")

    # Create DataFrame
    df = pd.DataFrame([
        {
            "Model": r.model_name,
            "Mode": r.mode,
            "Params (M)": r.params_millions,
            "Throughput (img/s)": r.throughput_img_per_sec,
            "Memory (MB/img)": r.memory_mb_per_image,
        }
        for r in results
    ])

    # Pivot table for comparison
    pivot_df = df.pivot(index="Model", columns="Mode", values="Throughput (img/s)")
    pivot_df = pivot_df[MODES]  # Reorder columns

    # Add speedup column
    pivot_df["Speedup (ac+c vs base)"] = pivot_df["autocast+compile"] / pivot_df["baseline"]
    pivot_df = pivot_df.sort_values("autocast+compile", ascending=False)

    print("\n" + "=" * 70)
    print("THROUGHPUT PAR MODE (img/s)")
    print("=" * 70)
    print(tabulate(pivot_df.reset_index(), headers="keys", tablefmt="grid", floatfmt=".1f", showindex=False))

    # Save results
    df.to_csv("benchmark_results_all.csv", index=False)
    pivot_df.to_csv("benchmark_results_comparison.csv")
    print("\nRésultats sauvegardés")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Throughput by mode
    ax1 = axes[0]
    x = range(len(pivot_df))
    width = 0.2

    for i, mode in enumerate(MODES):
        ax1.barh([xi + i * width for xi in x], pivot_df[mode], width, label=mode, alpha=0.8)

    ax1.set_yticks([xi + 1.5 * width for xi in x])
    ax1.set_yticklabels([m.replace("mobilenetv4_conv_", "mnv4_").replace(".e2400_r224_in1k", "").replace(".e500_r224_in1k", "") for m in pivot_df.index], fontsize=8)
    ax1.set_xlabel("Throughput (img/s)")
    ax1.set_title("Throughput par mode d'optimisation")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3, axis="x")

    # Plot 2: Speedup
    ax2 = axes[1]
    colors = ["green" if s > 1.5 else "orange" if s > 1.2 else "red" for s in pivot_df["Speedup (ac+c vs base)"]]
    ax2.barh(range(len(pivot_df)), pivot_df["Speedup (ac+c vs base)"], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(pivot_df)))
    ax2.set_yticklabels([m.replace("mobilenetv4_conv_", "mnv4_").replace(".e2400_r224_in1k", "").replace(".e500_r224_in1k", "") for m in pivot_df.index], fontsize=8)
    ax2.set_xlabel("Speedup (autocast+compile vs baseline)")
    ax2.set_title("Gain de performance")
    ax2.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("benchmark_plot.png", dpi=150)
    print("Graphique sauvegardé")

    # Summary stats
    print("\n" + "=" * 70)
    print("STATISTIQUES DE SPEEDUP (autocast+compile vs baseline)")
    print("=" * 70)
    speedups = pivot_df["Speedup (ac+c vs base)"]
    print(f"Speedup moyen: {speedups.mean():.2f}x")
    print(f"Speedup min: {speedups.min():.2f}x ({speedups.idxmin()})")
    print(f"Speedup max: {speedups.max():.2f}x ({speedups.idxmax()})")


if __name__ == "__main__":
    main()
