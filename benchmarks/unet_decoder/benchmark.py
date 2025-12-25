#!/usr/bin/env python3
"""Benchmark script for encoder + UNet-like decoder on CPU."""

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


# Encoders to benchmark (best performers from encoder-only benchmark)
ENCODERS = [
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s2",
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
    "edgenext_x_small",
    "edgenext_small",
    "resnet18",
    "resnet34",
    "repvgg_a0",
    "repvgg_a1",
    "efficientnet_b0",
    "efficientnet_b1",
    "ghostnetv2_100",
    "tinynet_b",
    "tinynet_c",
]


class ConvBlock(nn.Module):
    """Double convolution block for UNet decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDecoder(nn.Module):
    """UNet-like decoder with skip connections."""

    def __init__(self, encoder_channels, num_classes=1, decoder_channels=(256, 128, 64, 32)):
        super().__init__()

        # Reverse encoder channels for decoder (we go from deep to shallow)
        encoder_channels = encoder_channels[::-1]

        self.blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        in_ch = encoder_channels[0]
        for i, (enc_ch, dec_ch) in enumerate(zip(encoder_channels[1:], decoder_channels)):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, dec_ch, kernel_size=2, stride=2)
            )
            self.blocks.append(
                ConvBlock(dec_ch + enc_ch, dec_ch)
            )
            in_ch = dec_ch

        # Final upsampling to original size
        self.final_upconv = nn.ConvTranspose2d(
            decoder_channels[-1], decoder_channels[-1], kernel_size=2, stride=2
        )

        # Segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(self, features):
        # features: list from encoder [shallow, ..., deep]
        # Reverse to process from deep to shallow
        features = features[::-1]

        x = features[0]
        for i, (upconv, block) in enumerate(zip(self.upconvs, self.blocks)):
            x = upconv(x)
            skip = features[i + 1]

            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = self.final_upconv(x)
        return self.seg_head(x)


class SegmentationModel(nn.Module):
    """Encoder + UNet decoder for segmentation."""

    def __init__(self, encoder_name, num_classes=1):
        super().__init__()

        # Create encoder with feature extraction
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),  # Get all intermediate features
        )

        # Get feature channels from encoder
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        encoder_channels = [f.shape[1] for f in features]

        # Decoder channels adapted to encoder
        max_ch = max(encoder_channels)
        decoder_channels = [
            min(256, max_ch),
            min(128, max_ch // 2),
            min(64, max_ch // 4),
            min(32, max_ch // 8),
        ]

        self.decoder = UNetDecoder(encoder_channels, num_classes, decoder_channels)
        self.input_size = 512

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        # Ensure output matches input size
        if out.shape[-2:] != (self.input_size, self.input_size):
            out = F.interpolate(out, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
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
    """Benchmark encoder + UNet decoder."""

    input_size = 512

    # Create segmentation model
    try:
        model = SegmentationModel(encoder_name, num_classes=1)
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

    # Create dummy input
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
    print("BENCHMARK ENCODER + UNET DECODER sur CPU")
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

    # Create DataFrame
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
    print("\nRésultats sauvegardés dans benchmark_results.csv")

    # Create plot
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
    ax.set_title("CPU Throughput: Encoder + UNet Decoder @ 512×512\n(point size = memory, color = encoder params)", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Encoder Params (M)")

    plt.tight_layout()
    plt.savefig("benchmark_plot.png", dpi=150)
    print("Graphique sauvegardé dans benchmark_plot.png")

    print("\n" + "=" * 70)
    print("TOP 5 MODÈLES PAR EFFICACITÉ")
    print("=" * 70)
    top5 = df.head(5)[["Encoder", "Total Params (M)", "Throughput (img/s)", "Efficiency (img/s/M)"]]
    print(tabulate(top5, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False))


if __name__ == "__main__":
    main()
