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

# Models to benchmark - extended set up to ~40M params for comparison with STU-Net/TotalSegmentator
MODELS = [
    # MobileOne family (2M - 10M)
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s2",
    "mobileone_s3",
    # MobileNetV4 family (4M - 33M)
    "mobilenetv4_conv_small.e2400_r224_in1k",
    "mobilenetv4_conv_medium.e500_r224_in1k",
    "mobilenetv4_conv_large.e600_r384_in1k",
    # MobileNetV3 family (2.5M - 5.5M)
    "mobilenetv3_small_100",
    "mobilenetv3_large_100",
    # MobileNetV2 family (3.5M - 6M)
    "mobilenetv2_100",
    "mobilenetv2_140",
    # RegNetX family (2.7M - 15M)
    "regnetx_002",
    "regnetx_004",
    "regnetx_008",
    "regnetx_016",
    "regnetx_032",
    # RegNetY family (3M - 11M)
    "regnety_002",
    "regnety_004",
    "regnety_008",
    "regnety_016",
    # ResNet family (12M - 26M)
    "resnet18",
    "resnet34",
    "resnet50",
    # EfficientNet family (5M - 9M)
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    # RepVGG family (8M - 25M)
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "repvgg_b0",
    # LCNet family (3M - 4.5M)
    "lcnet_100",
    "lcnet_150",
    # TinyNet family (2.5M - 6M)
    "tinynet_a",
    "tinynet_b",
    "tinynet_c",
    # EdgeNeXt family (2M - 6M)
    "edgenext_x_small",
    "edgenext_small",
    # ConvNeXt family (29M - 50M)
    "convnext_tiny",
    "convnext_small",
    # GhostNetV2 family (6M - 12M)
    "ghostnetv2_100",
    "ghostnetv2_130",
    "ghostnetv2_160",
    # DenseNet family (8M - 14M)
    "densenet121",
    "densenet169",
    # ViT family (6M - 22M)
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_small_patch32_224",
    # PoolFormer family (12M - 21M)
    "poolformer_s12",
    "poolformer_s24",
]

# Model families for plotting
MODEL_FAMILIES = {
    "MobileOne": ["mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3"],
    "MobileNetV4": ["mobilenetv4_conv_small.e2400_r224_in1k", "mobilenetv4_conv_medium.e500_r224_in1k", "mobilenetv4_conv_large.e600_r384_in1k"],
    "MobileNetV3": ["mobilenetv3_small_100", "mobilenetv3_large_100"],
    "MobileNetV2": ["mobilenetv2_100", "mobilenetv2_140"],
    "RegNetX": ["regnetx_002", "regnetx_004", "regnetx_008", "regnetx_016", "regnetx_032"],
    "RegNetY": ["regnety_002", "regnety_004", "regnety_008", "regnety_016"],
    "ResNet": ["resnet18", "resnet34", "resnet50"],
    "EfficientNet": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
    "RepVGG": ["repvgg_a0", "repvgg_a1", "repvgg_a2", "repvgg_b0"],
    "LCNet": ["lcnet_100", "lcnet_150"],
    "TinyNet": ["tinynet_a", "tinynet_b", "tinynet_c"],
    "EdgeNeXt": ["edgenext_x_small", "edgenext_small"],
    "ConvNeXt": ["convnext_tiny", "convnext_small"],
    "GhostNetV2": ["ghostnetv2_100", "ghostnetv2_130", "ghostnetv2_160"],
    "DenseNet": ["densenet121", "densenet169"],
    "ViT": ["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_small_patch32_224"],
    "PoolFormer": ["poolformer_s12", "poolformer_s24"],
}

def get_model_family(model_name):
    """Get the family name for a model."""
    for family, models in MODEL_FAMILIES.items():
        if model_name in models:
            return family
    return "Other"

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

    # Create comparison plot (existing)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Throughput by mode
    ax1 = axes[0]
    x = range(len(pivot_df))
    width = 0.2

    for i, mode in enumerate(MODES):
        ax1.barh([xi + i * width for xi in x], pivot_df[mode], width, label=mode, alpha=0.8)

    ax1.set_yticks([xi + 1.5 * width for xi in x])
    ax1.set_yticklabels([m.replace("mobilenetv4_conv_", "mnv4_").replace(".e2400_r224_in1k", "").replace(".e500_r224_in1k", "").replace(".e600_r384_in1k", "") for m in pivot_df.index], fontsize=8)
    ax1.set_xlabel("Throughput (img/s)")
    ax1.set_title("Throughput par mode d'optimisation")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3, axis="x")

    # Plot 2: Speedup
    ax2 = axes[1]
    colors = ["green" if s > 1.5 else "orange" if s > 1.2 else "red" for s in pivot_df["Speedup (ac+c vs base)"]]
    ax2.barh(range(len(pivot_df)), pivot_df["Speedup (ac+c vs base)"], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(pivot_df)))
    ax2.set_yticklabels([m.replace("mobilenetv4_conv_", "mnv4_").replace(".e2400_r224_in1k", "").replace(".e500_r224_in1k", "").replace(".e600_r384_in1k", "") for m in pivot_df.index], fontsize=8)
    ax2.set_xlabel("Speedup (autocast+compile vs baseline)")
    ax2.set_title("Gain de performance")
    ax2.axvline(x=1.0, color="black", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("benchmark_plot.png", dpi=150)
    print("Graphique benchmark_plot.png sauvegardé")

    # NEW: Create scatter plot with model families
    # Get best results (autocast+compile mode)
    best_df = df[df["Mode"] == "autocast+compile"].copy()
    best_df["Family"] = best_df["Model"].apply(get_model_family)

    # Color palette for families
    family_colors = {
        "MobileOne": "#e41a1c",
        "MobileNetV4": "#377eb8",
        "MobileNetV3": "#4daf4a",
        "MobileNetV2": "#984ea3",
        "RegNetX": "#ff7f00",
        "RegNetY": "#ffff33",
        "ResNet": "#a65628",
        "EfficientNet": "#f781bf",
        "RepVGG": "#999999",
        "LCNet": "#66c2a5",
        "TinyNet": "#fc8d62",
        "EdgeNeXt": "#8da0cb",
        "ConvNeXt": "#e78ac3",
        "GhostNetV2": "#a6d854",
        "DenseNet": "#ffd92f",
        "ViT": "#e5c494",
        "PoolFormer": "#b3b3b3",
        "Other": "#000000",
    }

    fig2, ax = plt.subplots(figsize=(14, 10))

    # Normalize memory for point sizes
    mem_min, mem_max = best_df["Memory (MB/img)"].min(), best_df["Memory (MB/img)"].max()
    sizes = 50 + 400 * (best_df["Memory (MB/img)"] - mem_min) / (mem_max - mem_min + 1e-6)

    # Plot each family with lines connecting models
    for family in MODEL_FAMILIES.keys():
        family_data = best_df[best_df["Family"] == family].sort_values("Params (M)")
        if len(family_data) == 0:
            continue

        color = family_colors.get(family, "#000000")
        family_sizes = sizes[family_data.index]

        # Plot points
        ax.scatter(
            family_data["Params (M)"],
            family_data["Throughput (img/s)"],
            s=family_sizes,
            c=color,
            label=family,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        # Draw lines connecting models of the same family
        if len(family_data) > 1:
            ax.plot(
                family_data["Params (M)"],
                family_data["Throughput (img/s)"],
                c=color,
                alpha=0.4,
                linewidth=2,
                zorder=2,
            )

    # Add model labels
    for _, row in best_df.iterrows():
        label = (row["Model"]
            .replace("mobilenetv4_conv_", "mnv4_")
            .replace(".e2400_r224_in1k", "")
            .replace(".e500_r224_in1k", "")
            .replace(".e600_r384_in1k", "")
        )
        ax.annotate(
            label,
            (row["Params (M)"], row["Throughput (img/s)"]),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Parameters (Millions)", fontsize=12)
    ax.set_ylabel("Throughput (images/second)", fontsize=12)
    ax.set_title("CPU Throughput vs Parameters @ 512x512\n(point size = memory, color = model family, lines connect same family)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig("benchmark_scatter_families.png", dpi=150)
    print("Graphique benchmark_scatter_families.png sauvegardé")

    # Save best mode results for comparison
    best_df["Efficiency (img/s/M)"] = best_df["Throughput (img/s)"] / best_df["Params (M)"]
    best_df = best_df.sort_values("Efficiency (img/s/M)", ascending=False)
    best_df[["Model", "Params (M)", "Throughput (img/s)", "Memory (MB/img)", "Family", "Efficiency (img/s/M)"]].to_csv(
        "benchmark_results.csv", index=False
    )
    print("Résultats benchmark_results.csv sauvegardés")

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
