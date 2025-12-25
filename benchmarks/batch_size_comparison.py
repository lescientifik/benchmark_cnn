#!/usr/bin/env python3
"""Compare throughput at different batch sizes (1 vs 16)."""

import gc
import os
import time

import pandas as pd
import psutil
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

ENCODERS = [
    "mobileone_s0",
    "mobilenetv4_conv_small.e2400_r224_in1k",
    "regnetx_002",
    "regnetx_004",
    "lcnet_100",
    "mobilenetv3_small_100",
    "resnet18",
    "repvgg_a0",
]

BATCH_SIZES = [1, 4, 8, 16]
INPUT_SIZE = 512
FPN_CHANNELS = 128


class SemanticFPNDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes=1, fpn_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, fpn_channels, 1) for ch in encoder_channels])
        self.smooth_convs = nn.ModuleList([nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1) for _ in encoder_channels])
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i + 1], size=laterals[i].shape[-2:], mode="nearest")
        p2 = self.smooth_convs[0](laterals[0])
        return self.seg_head(p2)


class FPNSegModel(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        with torch.no_grad():
            features = self.encoder(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE))
        self.decoder = SemanticFPNDecoder([f.shape[1] for f in features], fpn_channels=FPN_CHANNELS)

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def benchmark(encoder_name: str, batch_size: int, num_warmup: int = 3, num_iterations: int = 30) -> dict:
    model = FPNSegModel(encoder_name)
    model.eval()

    if any(x in encoder_name for x in ["repvgg", "mobileone"]):
        for m in model.modules():
            if hasattr(m, "reparameterize"):
                m.reparameterize()

    model = torch.compile(model)
    dummy_input = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE)

    # Warmup
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Benchmark
    gc.collect()
    times = []
    with torch.inference_mode(), torch.autocast(device_type="cpu"):
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    throughput = batch_size / avg_time
    latency_per_img = avg_time / batch_size * 1000  # ms

    del model, dummy_input
    gc.collect()

    return {
        "encoder": encoder_name,
        "batch_size": batch_size,
        "throughput": throughput,
        "latency_ms": latency_per_img,
    }


def main():
    print("=" * 70)
    print("COMPARAISON BATCH SIZE (Semantic FPN)")
    print("=" * 70)
    print(f"Batch sizes: {BATCH_SIZES}, Input: {INPUT_SIZE}x{INPUT_SIZE}")
    print()

    results = []
    for encoder in tqdm(ENCODERS, desc="Encoders"):
        for bs in BATCH_SIZES:
            try:
                r = benchmark(encoder, bs)
                results.append(r)
                tqdm.write(f"  {encoder} bs={bs}: {r['throughput']:.1f} img/s, {r['latency_ms']:.1f} ms/img")
            except Exception as e:
                tqdm.write(f"  ERREUR {encoder} bs={bs}: {e}")

    df = pd.DataFrame(results)

    # Pivot pour comparaison
    pivot = df.pivot(index="encoder", columns="batch_size", values="throughput")
    pivot["speedup_16_vs_1"] = pivot[16] / pivot[1]
    pivot = pivot.sort_values(16, ascending=False)

    print("\n" + "=" * 70)
    print("THROUGHPUT (img/s) PAR BATCH SIZE")
    print("=" * 70)
    print(tabulate(pivot.reset_index(), headers="keys", tablefmt="grid", floatfmt=".1f", showindex=False))

    df.to_csv("batch_size_comparison.csv", index=False)
    print("\nRésultats sauvegardés: batch_size_comparison.csv")

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Speedup moyen bs=16 vs bs=1: {pivot['speedup_16_vs_1'].mean():.2f}x")
    print(f"Meilleur throughput bs=16: {pivot[16].max():.1f} img/s ({pivot[16].idxmax()})")


if __name__ == "__main__":
    main()
