#!/usr/bin/env python3
"""Benchmark script for encoder + Semantic FPN decoder on CPU.

Uses standard Semantic FPN architecture as described in:
- "Panoptic Feature Pyramid Networks" (Kirillov et al., 2019)
- Uses only the highest resolution FPN level (P2) for segmentation
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
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm


# Encoders to benchmark - extended set up to ~40M params
ENCODERS = [
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
    "PoolFormer": ["poolformer_s12", "poolformer_s24"],
}


def get_model_family(model_name):
    """Get the family name for a model."""
    for family, models in MODEL_FAMILIES.items():
        if model_name in models:
            return family
    return "Other"


class SemanticFPNDecoder(nn.Module):
    """Semantic FPN decoder - standard implementation.

    Architecture:
    - Lateral connections: 1x1 conv to reduce channels to fpn_channels
    - Top-down pathway: upsample + add
    - Uses only P2 (highest resolution) for final prediction
    - Final upsample to input resolution

    Reference: Panoptic FPN (Kirillov et al., 2019)
    """

    def __init__(self, encoder_channels, num_classes=1, fpn_channels=128):
        super().__init__()
        self.fpn_channels = fpn_channels

        # Lateral connections (1x1 convs)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1) for ch in encoder_channels
        ])

        # Smooth convs after fusion (3x3 convs)
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
            for _ in encoder_channels
        ])

        # Segmentation head on P2 only
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

    def forward(self, features):
        # features: [C2, C3, C4, C5] from encoder (low to high level)

        # Build laterals
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway (from P5 to P2)
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest"
            )

        # Apply smooth convs
        fpn_features = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]

        # Use P2 (highest resolution) for segmentation
        p2 = fpn_features[0]

        return self.seg_head(p2)


class SegmentationModel(nn.Module):
    """Encoder + Semantic FPN decoder."""

    def __init__(self, encoder_name, num_classes=1, fpn_channels=128):
        super().__init__()
        self.input_size = 512

        # Create encoder with feature extraction
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # C2, C3, C4, C5
        )

        # Get feature channels from encoder
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        encoder_channels = [f.shape[1] for f in features]
        self.feature_stride = self.input_size // features[0].shape[-1]  # Usually 4

        self.decoder = SemanticFPNDecoder(encoder_channels, num_classes, fpn_channels)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)

        # Final upsample to input resolution
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


@dataclass
class BenchmarkResult:
    model_name: str
    params_millions: float
    encoder_params_millions: float
    throughput_img_per_sec: float
    memory_mb_per_image: float
    input_size: int


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_model(
    encoder_name: str,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> BenchmarkResult:
    """Benchmark encoder + Semantic FPN decoder."""

    input_size = 512

    try:
        model = SegmentationModel(encoder_name, num_classes=1, fpn_channels=128)
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}")

    model.eval()

    # Reparameterize if applicable
    if any(x in encoder_name for x in ["repvgg", "mobileone", "fastvit"]):
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()

    # Compile model
    model = torch.compile(model)

    # Count parameters
    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    total_params_m = total_params / 1e6
    encoder_params_m = encoder_params / 1e6

    dummy_input = torch.randn(batch_size, 3, input_size, input_size)

    # Warmup
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Measure memory
    model_memory_mb = (total_params * 4) / (1024 * 1024)
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    peak_mem = mem_before
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(3):
            _ = model(dummy_input)
            current_mem = process.memory_info().rss / (1024 * 1024)
            peak_mem = max(peak_mem, current_mem)

    activation_memory = max(0, peak_mem - mem_before)
    memory_per_image = model_memory_mb + activation_memory / batch_size

    # Measure throughput
    gc.collect()
    times = []

    with torch.inference_mode(), torch.autocast(device_type="cpu"):
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
        model_name=encoder_name,
        params_millions=total_params_m,
        encoder_params_millions=encoder_params_m,
        throughput_img_per_sec=throughput,
        memory_mb_per_image=memory_per_image,
        input_size=input_size,
    )


def main():
    print("=" * 70)
    print("BENCHMARK ENCODER + SEMANTIC FPN DECODER sur CPU")
    print("=" * 70)
    print(f"Nombre d'encodeurs à tester: {len(ENCODERS)}")
    print()

    results = []

    for encoder_name in tqdm(ENCODERS, desc="Benchmarking"):
        try:
            result = benchmark_model(encoder_name)
            results.append(result)
            tqdm.write(
                f"  {encoder_name}: {result.throughput_img_per_sec:.1f} img/s, "
                f"{result.params_millions:.1f}M params (enc: {result.encoder_params_millions:.1f}M)"
            )
        except Exception as e:
            tqdm.write(f"  ERREUR {encoder_name}: {e}")

    df = pd.DataFrame([
        {
            "Encoder": r.model_name,
            "Total Params (M)": r.params_millions,
            "Encoder Params (M)": r.encoder_params_millions,
            "Throughput (img/s)": r.throughput_img_per_sec,
            "Memory (MB/img)": r.memory_mb_per_image,
            "Efficiency (img/s/M)": r.throughput_img_per_sec / r.params_millions,
        }
        for r in results
    ])

    df = df.sort_values("Efficiency (img/s/M)", ascending=False)

    print("\n" + "=" * 70)
    print("RÉSULTATS (triés par efficacité)")
    print("=" * 70)
    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False))

    df.to_csv("benchmark_results.csv", index=False)
    print("\nRésultats sauvegardés")

    # Create original plot (keep existing)
    fig, ax = plt.subplots(figsize=(12, 8))

    mem_min, mem_max = df["Memory (MB/img)"].min(), df["Memory (MB/img)"].max()
    sizes = 50 + 300 * (df["Memory (MB/img)"] - mem_min) / (mem_max - mem_min + 1e-6)

    scatter = ax.scatter(
        df["Total Params (M)"],
        df["Throughput (img/s)"],
        s=sizes,
        alpha=0.7,
        c=df["Encoder Params (M)"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.5,
    )

    for _, row in df.iterrows():
        label = (row["Encoder"]
            .replace("mobilenetv4_conv_", "mnv4_")
            .replace(".e2400_r224_in1k", "")
            .replace(".e500_r224_in1k", "")
            .replace(".e600_r384_in1k", "")
        )
        ax.annotate(
            label,
            (row["Total Params (M)"], row["Throughput (img/s)"]),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Total Parameters (Millions)", fontsize=12)
    ax.set_ylabel("Throughput (images/second)", fontsize=12)
    ax.set_title("CPU Throughput: Encoder + Semantic FPN @ 512x512\n(point size = memory, color = encoder params)", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Encoder Params (M)")

    plt.tight_layout()
    plt.savefig("benchmark_plot.png", dpi=150)
    print("Graphique benchmark_plot.png sauvegardé")

    # NEW: Create scatter plot with model families
    df["Family"] = df["Encoder"].apply(get_model_family)

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
        "PoolFormer": "#b3b3b3",
        "Other": "#000000",
    }

    fig2, ax2 = plt.subplots(figsize=(14, 10))

    # Normalize memory for point sizes
    sizes2 = 50 + 400 * (df["Memory (MB/img)"] - mem_min) / (mem_max - mem_min + 1e-6)

    # Plot each family with lines connecting models
    for family in MODEL_FAMILIES.keys():
        family_data = df[df["Family"] == family].sort_values("Total Params (M)")
        if len(family_data) == 0:
            continue

        color = family_colors.get(family, "#000000")
        family_sizes = sizes2[family_data.index]

        # Plot points
        ax2.scatter(
            family_data["Total Params (M)"],
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
            ax2.plot(
                family_data["Total Params (M)"],
                family_data["Throughput (img/s)"],
                c=color,
                alpha=0.4,
                linewidth=2,
                zorder=2,
            )

    # Add model labels
    for _, row in df.iterrows():
        label = (row["Encoder"]
            .replace("mobilenetv4_conv_", "mnv4_")
            .replace(".e2400_r224_in1k", "")
            .replace(".e500_r224_in1k", "")
            .replace(".e600_r384_in1k", "")
        )
        ax2.annotate(
            label,
            (row["Total Params (M)"], row["Throughput (img/s)"]),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax2.set_xlabel("Total Parameters (Millions)", fontsize=12)
    ax2.set_ylabel("Throughput (images/second)", fontsize=12)
    ax2.set_title("CPU Throughput: Encoder + Semantic FPN @ 512x512\n(point size = memory, color = model family, lines connect same family)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig("benchmark_scatter_families.png", dpi=150)
    print("Graphique benchmark_scatter_families.png sauvegardé")

    print("\n" + "=" * 70)
    print("TOP 5 MODÈLES PAR EFFICACITÉ")
    print("=" * 70)
    top5 = df.head(5)[["Encoder", "Total Params (M)", "Throughput (img/s)", "Efficiency (img/s/M)"]]
    print(tabulate(top5, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False))


if __name__ == "__main__":
    main()
