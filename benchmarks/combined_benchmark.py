#!/usr/bin/env python3
"""Combined benchmark comparing FPN vs UNet decoders side by side.

Creates a paneled plot:
- Top left: FPN throughput vs params
- Top right: UNet throughput vs params
- Bottom left: FPN decoder size per encoder
- Bottom right: UNet decoder size per encoder
"""

import gc
import os
import sys
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

# Add parent directories to path
sys.path.insert(0, os.path.dirname(__file__))

ENCODERS = [
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
]

INPUT_SIZE = 512
FPN_CHANNELS = 128


# ============================================================================
# Semantic FPN Decoder (from fpn_decoder/benchmark.py)
# ============================================================================

class SemanticFPNDecoder(nn.Module):
    """Semantic FPN decoder - uses only P2 for segmentation."""

    def __init__(self, encoder_channels, num_classes=1, fpn_channels=128):
        super().__init__()
        self.fpn_channels = fpn_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, fpn_channels, 1) for ch in encoder_channels
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
            for _ in encoder_channels
        ])
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest"
            )
        fpn_features = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]
        p2 = fpn_features[0]
        return self.seg_head(p2)


# ============================================================================
# UNet Decoder (from unet_decoder/benchmark.py)
# ============================================================================

class ConvBlock(nn.Module):
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
    def __init__(self, encoder_channels, num_classes=1, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        self.blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        in_ch = encoder_channels[0]
        for i, (enc_ch, dec_ch) in enumerate(zip(encoder_channels[1:], decoder_channels)):
            self.upconvs.append(nn.ConvTranspose2d(in_ch, dec_ch, kernel_size=2, stride=2))
            self.blocks.append(ConvBlock(dec_ch + enc_ch, dec_ch))
            in_ch = dec_ch
        self.final_upconv = nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1], kernel_size=2, stride=2)
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        for i, (upconv, block) in enumerate(zip(self.upconvs, self.blocks)):
            x = upconv(x)
            skip = features[i + 1]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        x = self.final_upconv(x)
        return self.seg_head(x)


# ============================================================================
# Segmentation Models
# ============================================================================

class FPNSegModel(nn.Module):
    def __init__(self, encoder_name, num_classes=1, fpn_channels=128):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4)
        )
        dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        encoder_channels = [f.shape[1] for f in features]
        self.decoder = SemanticFPNDecoder(encoder_channels, num_classes, fpn_channels)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class UNetSegModel(nn.Module):
    def __init__(self, encoder_name, num_classes=1):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3, 4)
        )
        dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        encoder_channels = [f.shape[1] for f in features]
        max_ch = max(encoder_channels)
        decoder_channels = [min(256, max_ch), min(128, max_ch // 2), min(64, max_ch // 4), min(32, max_ch // 8)]
        self.decoder = UNetDecoder(encoder_channels, num_classes, decoder_channels)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# ============================================================================
# Benchmark Functions
# ============================================================================

@dataclass
class BenchmarkResult:
    encoder_name: str
    decoder_type: str
    total_params_m: float
    encoder_params_m: float
    decoder_params_m: float
    throughput: float
    memory_mb: float


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def reparameterize_model(model: nn.Module, encoder_name: str):
    """Apply reparameterization for RepVGG, MobileOne, etc."""
    if any(x in encoder_name for x in ["repvgg", "mobileone", "fastvit"]):
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()


def benchmark_model(
    model: nn.Module,
    encoder_name: str,
    decoder_type: str,
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> BenchmarkResult:
    """Benchmark a segmentation model."""

    model.eval()
    reparameterize_model(model, encoder_name)
    model = torch.compile(model)

    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Warmup
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Memory
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    peak_mem = mem_before
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(3):
            _ = model(dummy_input)
            peak_mem = max(peak_mem, process.memory_info().rss / (1024 * 1024))
    memory_mb = (total_params * 4) / (1024 * 1024) + max(0, peak_mem - mem_before)

    # Throughput
    gc.collect()
    times = []
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            times.append(time.perf_counter() - start)

    throughput = 1.0 / (sum(times) / len(times))

    del model, dummy_input
    gc.collect()

    return BenchmarkResult(
        encoder_name=encoder_name,
        decoder_type=decoder_type,
        total_params_m=total_params / 1e6,
        encoder_params_m=encoder_params / 1e6,
        decoder_params_m=decoder_params / 1e6,
        throughput=throughput,
        memory_mb=memory_mb,
    )


def main():
    print("=" * 70)
    print("BENCHMARK COMBINÉ FPN vs UNET")
    print("=" * 70)
    print(f"Encoders: {len(ENCODERS)}, Input: {INPUT_SIZE}x{INPUT_SIZE}")
    print()

    results = []

    for encoder_name in tqdm(ENCODERS, desc="Benchmarking"):
        # FPN
        try:
            model = FPNSegModel(encoder_name, fpn_channels=FPN_CHANNELS)
            result = benchmark_model(model, encoder_name, "FPN")
            results.append(result)
            tqdm.write(f"  {encoder_name} [FPN]: {result.throughput:.1f} img/s, dec={result.decoder_params_m:.2f}M")
        except Exception as e:
            tqdm.write(f"  ERREUR {encoder_name} [FPN]: {e}")

        # UNet
        try:
            model = UNetSegModel(encoder_name)
            result = benchmark_model(model, encoder_name, "UNet")
            results.append(result)
            tqdm.write(f"  {encoder_name} [UNet]: {result.throughput:.1f} img/s, dec={result.decoder_params_m:.2f}M")
        except Exception as e:
            tqdm.write(f"  ERREUR {encoder_name} [UNet]: {e}")

    # Create DataFrame
    df = pd.DataFrame([vars(r) for r in results])
    df.to_csv("combined_benchmark_results.csv", index=False)

    # Split by decoder type
    df_fpn = df[df["decoder_type"] == "FPN"].copy()
    df_unet = df[df["decoder_type"] == "UNet"].copy()

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARAISON FPN vs UNET")
    print("=" * 70)

    comparison = df_fpn[["encoder_name", "throughput", "decoder_params_m"]].merge(
        df_unet[["encoder_name", "throughput", "decoder_params_m"]],
        on="encoder_name",
        suffixes=("_fpn", "_unet")
    )
    comparison["speedup_fpn"] = comparison["throughput_fpn"] / comparison["throughput_unet"]
    comparison = comparison.sort_values("throughput_fpn", ascending=False)
    print(tabulate(comparison, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False))

    # Create paneled plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Helper to shorten names
    def short_name(name):
        return (name.replace("mobilenetv4_conv_", "mnv4_")
                    .replace(".e2400_r224_in1k", "")
                    .replace(".e500_r224_in1k", ""))

    # Top left: FPN throughput vs params
    ax = axes[0, 0]
    ax.scatter(df_fpn["total_params_m"], df_fpn["throughput"], s=100, alpha=0.7, c="tab:blue")
    for _, row in df_fpn.iterrows():
        ax.annotate(short_name(row["encoder_name"]), (row["total_params_m"], row["throughput"]),
                    fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Total Params (M)")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Semantic FPN - Throughput vs Params")
    ax.grid(True, alpha=0.3)

    # Top right: UNet throughput vs params
    ax = axes[0, 1]
    ax.scatter(df_unet["total_params_m"], df_unet["throughput"], s=100, alpha=0.7, c="tab:orange")
    for _, row in df_unet.iterrows():
        ax.annotate(short_name(row["encoder_name"]), (row["total_params_m"], row["throughput"]),
                    fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Total Params (M)")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("UNet - Throughput vs Params")
    ax.grid(True, alpha=0.3)

    # Bottom left: FPN decoder size bar plot
    ax = axes[1, 0]
    df_fpn_sorted = df_fpn.sort_values("decoder_params_m")
    bars = ax.barh(range(len(df_fpn_sorted)), df_fpn_sorted["decoder_params_m"], color="tab:blue", alpha=0.7)
    ax.set_yticks(range(len(df_fpn_sorted)))
    ax.set_yticklabels([short_name(n) for n in df_fpn_sorted["encoder_name"]], fontsize=8)
    ax.set_xlabel("Decoder Params (M)")
    ax.set_title("FPN Decoder Size")
    ax.grid(True, alpha=0.3, axis="x")

    # Bottom right: UNet decoder size bar plot
    ax = axes[1, 1]
    df_unet_sorted = df_unet.sort_values("decoder_params_m")
    bars = ax.barh(range(len(df_unet_sorted)), df_unet_sorted["decoder_params_m"], color="tab:orange", alpha=0.7)
    ax.set_yticks(range(len(df_unet_sorted)))
    ax.set_yticklabels([short_name(n) for n in df_unet_sorted["encoder_name"]], fontsize=8)
    ax.set_xlabel("Decoder Params (M)")
    ax.set_title("UNet Decoder Size")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("combined_benchmark_plot.png", dpi=150)
    print("\nGraphique sauvegardé: combined_benchmark_plot.png")

    # Summary
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"FPN moyen: {df_fpn['throughput'].mean():.1f} img/s, decoder: {df_fpn['decoder_params_m'].mean():.2f}M")
    print(f"UNet moyen: {df_unet['throughput'].mean():.1f} img/s, decoder: {df_unet['decoder_params_m'].mean():.2f}M")
    print(f"Speedup FPN vs UNet: {df_fpn['throughput'].mean() / df_unet['throughput'].mean():.2f}x")


if __name__ == "__main__":
    main()
