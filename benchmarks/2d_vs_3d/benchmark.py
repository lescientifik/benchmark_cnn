#!/usr/bin/env python3
"""Comparison: 2D models (FPN/UNet) vs 3D models (STU-Net/TotalSegmentator).

This script benchmarks 2D segmentation models against 3D medical imaging models
to compare throughput at similar parameter counts.
"""

import gc
import statistics
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch._dynamo
from tqdm import tqdm

# Import shared models
from models import SegmentationModel, create_stunet_s, create_totalsegmentator

NUM_CLASSES = 117


# 2D encoders to benchmark (focused on RepVGG, RegNet, MobileNetV4, ResNet)
ENCODERS_2D = [
    # MobileNetV4 family (4M - 33M) - efficient modern architecture
    ("mnv4_small", "mobilenetv4_conv_small.e2400_r224_in1k"),
    ("mnv4_medium", "mobilenetv4_conv_medium.e500_r224_in1k"),
    ("mnv4_large", "mobilenetv4_conv_large.e600_r384_in1k"),
    # ResNet family (12M - 26M) - proven baseline
    ("resnet18", "resnet18"),
    ("resnet34", "resnet34"),
    ("resnet50", "resnet50"),
    # RegNetX family (2.7M - 15M) - efficient scaling
    ("regnetx_002", "regnetx_002"),
    ("regnetx_004", "regnetx_004"),
    ("regnetx_008", "regnetx_008"),
    ("regnetx_016", "regnetx_016"),
    ("regnetx_032", "regnetx_032"),
    # RegNetY family (3M - 11M) - with SE blocks
    ("regnety_002", "regnety_002"),
    ("regnety_004", "regnety_004"),
    ("regnety_008", "regnety_008"),
    ("regnety_016", "regnety_016"),
    # RepVGG family (8M - 55M) - reparameterizable for fast inference
    ("repvgg_a0", "repvgg_a0"),
    ("repvgg_a1", "repvgg_a1"),
    ("repvgg_a2", "repvgg_a2"),
    ("repvgg_b0", "repvgg_b0"),
    ("repvgg_b1", "repvgg_b1"),
    ("repvgg_b2", "repvgg_b2"),
    ("repvgg_b3", "repvgg_b3"),
]

DECODER_TYPES = ["fpn", "unet"]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark(model, input_tensor, warmup=10, iterations=50):
    """Benchmark model inference time with reduced variance.

    Key improvements over naive approach:
    - Reset dynamo cache before each model (avoids cross-contamination)
    - More warmup iterations (10 vs 3) for stable JIT compilation
    - More measurement iterations (50 vs 20) for statistical stability
    - Trimmed median (removes top/bottom 10% outliers)
    """
    # Reset dynamo cache to avoid cross-model interference
    torch._dynamo.reset()
    gc.collect()

    model.eval()

    # Try to compile
    try:
        model = torch.compile(model)
    except Exception:
        pass

    # Extended warmup for stable JIT compilation
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    times = []
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            times.append(time.perf_counter() - start)

    # Trimmed median: remove top/bottom 10% outliers
    times_sorted = sorted(times)
    trim = len(times) // 10
    times_trimmed = times_sorted[trim:-trim] if trim > 0 else times_sorted

    return statistics.median(times_trimmed)


def reparameterize_model(model, encoder_name):
    """Reparameterize RepVGG-style models for inference."""
    if "repvgg" in encoder_name:
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()


if __name__ == "__main__":
    print("=" * 85)
    print(f"COMPARISON: 2D Models vs 3D Models for {NUM_CLASSES}-class Segmentation")
    print("=" * 85)

    results = []

    # === 3D Models ===
    print("\n### 3D Models @ 128x128x64 ###")
    input_3d = torch.randn(1, 1, 128, 128, 64)
    equiv_slices = (128 * 128 * 64) / (512 * 512)  # Equivalent 512x512 images

    models_3d = [
        ("STU-Net-S", create_stunet_s),
        ("TotalSegmentator", create_totalsegmentator),
    ]

    for name, create_fn in tqdm(models_3d, desc="3D models"):
        try:
            model = create_fn(NUM_CLASSES)
            params = count_params(model)
            avg_time = benchmark(model, input_3d)
            throughput = 1.0 / avg_time
            equiv_tp = equiv_slices / avg_time

            results.append({
                "Model": name,
                "Encoder": "-",
                "Decoder": "3D",
                "Type": "3D",
                "Params (M)": params / 1e6,
                "Time (ms)": avg_time * 1000,
                "Throughput (vol/s)": throughput,
                "Equiv Throughput (img/s)": equiv_tp,
            })
            tqdm.write(f"  {name}: {params/1e6:.2f}M params, {avg_time*1000:.1f}ms, {equiv_tp:.1f} equiv/s")

            del model
            gc.collect()
        except Exception as e:
            tqdm.write(f"  {name}: ERROR - {e}")

    # === 2D Models ===
    print("\n### 2D Models @ 512x512 ###")
    input_2d = torch.randn(1, 3, 512, 512)

    total_2d = len(ENCODERS_2D) * len(DECODER_TYPES)
    pbar = tqdm(total=total_2d, desc="2D models")

    for display_name, encoder_name in ENCODERS_2D:
        for decoder_type in DECODER_TYPES:
            try:
                model = SegmentationModel(encoder_name, decoder_type, NUM_CLASSES)
                reparameterize_model(model, encoder_name)

                params = count_params(model)
                avg_time = benchmark(model, input_2d)
                throughput = 1.0 / avg_time

                full_name = f"{display_name} + {decoder_type.upper()}"
                results.append({
                    "Model": full_name,
                    "Encoder": display_name,
                    "Decoder": decoder_type.upper(),
                    "Type": "2D",
                    "Params (M)": params / 1e6,
                    "Time (ms)": avg_time * 1000,
                    "Throughput (vol/s)": throughput,
                    "Equiv Throughput (img/s)": throughput,  # 2D: 1 img = 1 vol
                })
                tqdm.write(f"  {full_name}: {params/1e6:.2f}M params, {avg_time*1000:.1f}ms, {throughput:.1f} img/s")

                del model
                gc.collect()
            except Exception as e:
                tqdm.write(f"  {display_name} + {decoder_type}: ERROR - {e}")

            pbar.update(1)

    pbar.close()

    # === Create DataFrame ===
    df = pd.DataFrame(results)
    df.to_csv("comparison_2d_vs_3d.csv", index=False)
    print("\nResults saved to comparison_2d_vs_3d.csv")

    # === Create comparison plot ===
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Get 3D baselines
    stunet_tp = df[df["Model"] == "STU-Net-S"]["Equiv Throughput (img/s)"].values[0]
    totalseg_tp = df[df["Model"] == "TotalSegmentator"]["Equiv Throughput (img/s)"].values[0]

    # Separate 2D models by decoder type
    df_2d = df[df["Type"] == "2D"].copy()
    df_fpn = df_2d[df_2d["Decoder"] == "FPN"].sort_values("Equiv Throughput (img/s)", ascending=True)
    df_unet = df_2d[df_2d["Decoder"] == "UNET"].sort_values("Equiv Throughput (img/s)", ascending=True)

    # Color by encoder family
    def get_color(encoder):
        if "mnv4" in encoder:
            return "#377eb8"  # Blue - MobileNetV4
        elif "resnet" in encoder:
            return "#a65628"  # Brown - ResNet
        elif "regnetx" in encoder:
            return "#ff7f00"  # Orange - RegNetX
        elif "regnety" in encoder:
            return "#ffff33"  # Yellow - RegNetY
        elif "repvgg" in encoder:
            return "#e41a1c"  # Red - RepVGG
        return "#999999"

    # Plot 1: FPN decoder
    ax1 = axes[0]
    colors_fpn = [get_color(e) for e in df_fpn["Encoder"]]
    y_pos = range(len(df_fpn))
    bars = ax1.barh(y_pos, df_fpn["Equiv Throughput (img/s)"], color=colors_fpn, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{m} ({p:.1f}M)" for m, p in zip(df_fpn["Encoder"], df_fpn["Params (M)"])], fontsize=9)
    ax1.set_xlabel("Throughput (img/s)", fontsize=12)
    ax1.set_title("2D Models with FPN Decoder\nvs 3D Models", fontsize=12)
    ax1.axvline(x=stunet_tp, color="green", linestyle="--", linewidth=2, label=f"STU-Net-S: {stunet_tp:.1f}")
    ax1.axvline(x=totalseg_tp, color="purple", linestyle="--", linewidth=2, label=f"TotalSeg: {totalseg_tp:.1f}")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3, axis="x")

    # Plot 2: UNet decoder
    ax2 = axes[1]
    colors_unet = [get_color(e) for e in df_unet["Encoder"]]
    y_pos = range(len(df_unet))
    ax2.barh(y_pos, df_unet["Equiv Throughput (img/s)"], color=colors_unet, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{m} ({p:.1f}M)" for m, p in zip(df_unet["Encoder"], df_unet["Params (M)"])], fontsize=9)
    ax2.set_xlabel("Throughput (img/s)", fontsize=12)
    ax2.set_title("2D Models with UNet Decoder\nvs 3D Models", fontsize=12)
    ax2.axvline(x=stunet_tp, color="green", linestyle="--", linewidth=2, label=f"STU-Net-S: {stunet_tp:.1f}")
    ax2.axvline(x=totalseg_tp, color="purple", linestyle="--", linewidth=2, label=f"TotalSeg: {totalseg_tp:.1f}")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    # Add legend for encoder families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#377eb8', label='MobileNetV4'),
        Patch(facecolor='#a65628', label='ResNet'),
        Patch(facecolor='#ff7f00', label='RegNetX'),
        Patch(facecolor='#ffff33', label='RegNetY'),
        Patch(facecolor='#e41a1c', label='RepVGG'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.02), fontsize=10)

    plt.suptitle(f"2D vs 3D Segmentation: Throughput Comparison\n(CPU, {NUM_CLASSES} classes, batch=1)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("comparison_2d_vs_3d.png", dpi=150, bbox_inches="tight")
    print("Plot saved to comparison_2d_vs_3d.png")

    # === Speedup summary ===
    print("\n" + "=" * 85)
    print("SPEEDUP SUMMARY vs STU-Net-S")
    print("=" * 85)

    df_2d_sorted = df_2d.sort_values("Equiv Throughput (img/s)", ascending=False)
    print(f"\nSTU-Net-S baseline: {stunet_tp:.1f} equiv img/s ({df[df['Model']=='STU-Net-S']['Params (M)'].values[0]:.1f}M params)")
    print(f"TotalSegmentator baseline: {totalseg_tp:.1f} equiv img/s ({df[df['Model']=='TotalSegmentator']['Params (M)'].values[0]:.1f}M params)")
    print()

    print("Top 10 fastest 2D models:")
    for i, (_, row) in enumerate(df_2d_sorted.head(10).iterrows()):
        speedup_stunet = row["Equiv Throughput (img/s)"] / stunet_tp
        speedup_totalseg = row["Equiv Throughput (img/s)"] / totalseg_tp
        print(f"  {i+1}. {row['Model']}: {row['Equiv Throughput (img/s)']:.1f} img/s "
              f"({speedup_stunet:.1f}x vs STU-Net, {speedup_totalseg:.1f}x vs TotalSeg) "
              f"- {row['Params (M)']:.1f}M params")

    # RepVGG focus
    print("\n" + "=" * 85)
    print("REPVGG FOCUS")
    print("=" * 85)
    df_repvgg = df_2d[df_2d["Encoder"].str.contains("repvgg")].sort_values("Equiv Throughput (img/s)", ascending=False)
    for _, row in df_repvgg.iterrows():
        speedup = row["Equiv Throughput (img/s)"] / stunet_tp
        print(f"  {row['Model']}: {row['Equiv Throughput (img/s)']:.1f} img/s ({speedup:.1f}x vs STU-Net) - {row['Params (M)']:.1f}M params")
