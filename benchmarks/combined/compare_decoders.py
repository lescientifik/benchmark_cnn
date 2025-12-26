#!/usr/bin/env python3
"""Compare FPN and UNet decoder benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
fpn_df = pd.read_csv("fpn_decoder/benchmark_results.csv")
unet_df = pd.read_csv("unet_decoder/benchmark_results.csv")

# Merge on encoder
merged = pd.merge(fpn_df, unet_df, on="Encoder", suffixes=("_fpn", "_unet"))

# Create figure with multiple panels
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("FPN vs UNet Decoder Comparison", fontsize=14, fontweight="bold")

# Panel 1: Throughput comparison scatter
ax1 = axes[0, 0]
ax1.scatter(merged["Throughput (img/s)_fpn"], merged["Throughput (img/s)_unet"],
            s=50, alpha=0.7, c=merged["Encoder Params (M)_fpn"], cmap="viridis")
# Add diagonal line (equal performance)
max_val = max(merged["Throughput (img/s)_fpn"].max(), merged["Throughput (img/s)_unet"].max())
ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal performance")
ax1.set_xlabel("FPN Throughput (img/s)")
ax1.set_ylabel("UNet Throughput (img/s)")
ax1.set_title("Throughput: FPN vs UNet")
cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
cbar1.set_label("Encoder Params (M)")
# Annotate top performers
for idx, row in merged.nlargest(5, "Throughput (img/s)_fpn").iterrows():
    ax1.annotate(row["Encoder"], (row["Throughput (img/s)_fpn"], row["Throughput (img/s)_unet"]),
                 fontsize=7, alpha=0.8)

# Panel 2: Memory comparison scatter
ax2 = axes[0, 1]
ax2.scatter(merged["Memory (MB/img)_fpn"], merged["Memory (MB/img)_unet"],
            s=50, alpha=0.7, c=merged["Encoder Params (M)_fpn"], cmap="viridis")
max_mem = max(merged["Memory (MB/img)_fpn"].max(), merged["Memory (MB/img)_unet"].max())
ax2.plot([0, max_mem], [0, max_mem], "k--", alpha=0.5, label="Equal memory")
ax2.set_xlabel("FPN Memory (MB/img)")
ax2.set_ylabel("UNet Memory (MB/img)")
ax2.set_title("Memory Usage: FPN vs UNet")
cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
cbar2.set_label("Encoder Params (M)")

# Panel 3: Efficiency bar chart (top 15 encoders by FPN efficiency)
ax3 = axes[1, 0]
top_encoders = merged.nlargest(15, "Efficiency (img/s/M)_fpn")
x = np.arange(len(top_encoders))
width = 0.35
bars1 = ax3.bar(x - width/2, top_encoders["Efficiency (img/s/M)_fpn"], width, label="FPN", color="steelblue")
bars2 = ax3.bar(x + width/2, top_encoders["Efficiency (img/s/M)_unet"], width, label="UNet", color="coral")
ax3.set_xlabel("Encoder")
ax3.set_ylabel("Efficiency (img/s/M params)")
ax3.set_title("Efficiency Comparison (Top 15)")
ax3.set_xticks(x)
ax3.set_xticklabels(top_encoders["Encoder"], rotation=45, ha="right", fontsize=8)
ax3.legend()

# Panel 4: Throughput vs Memory (Pareto frontier view)
ax4 = axes[1, 1]
# Plot both decoders
ax4.scatter(merged["Memory (MB/img)_fpn"], merged["Throughput (img/s)_fpn"],
            s=60, alpha=0.7, label="FPN", marker="o", c="steelblue")
ax4.scatter(merged["Memory (MB/img)_unet"], merged["Throughput (img/s)_unet"],
            s=60, alpha=0.7, label="UNet", marker="s", c="coral")

# Connect same encoders with lines
for idx, row in merged.iterrows():
    ax4.plot([row["Memory (MB/img)_fpn"], row["Memory (MB/img)_unet"]],
             [row["Throughput (img/s)_fpn"], row["Throughput (img/s)_unet"]],
             "gray", alpha=0.3, linewidth=0.8)

# Annotate Pareto-optimal points for FPN
# Simple Pareto: higher throughput AND lower memory is better
pareto_fpn = []
for idx, row in merged.iterrows():
    is_dominated = False
    for idx2, row2 in merged.iterrows():
        if (row2["Throughput (img/s)_fpn"] > row["Throughput (img/s)_fpn"] and
            row2["Memory (MB/img)_fpn"] < row["Memory (MB/img)_fpn"]):
            is_dominated = True
            break
    if not is_dominated:
        pareto_fpn.append(row["Encoder"])
        ax4.annotate(row["Encoder"],
                    (row["Memory (MB/img)_fpn"], row["Throughput (img/s)_fpn"]),
                    fontsize=7, alpha=0.9, fontweight="bold")

ax4.set_xlabel("Memory (MB/img)")
ax4.set_ylabel("Throughput (img/s)")
ax4.set_title("Throughput vs Memory (lines connect same encoder)")
ax4.legend()

plt.tight_layout()
plt.savefig("decoder_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Create a summary table
print("\n" + "="*80)
print("DECODER COMPARISON SUMMARY")
print("="*80)

# Calculate ratios
merged["throughput_ratio"] = merged["Throughput (img/s)_fpn"] / merged["Throughput (img/s)_unet"]
merged["memory_ratio"] = merged["Memory (MB/img)_fpn"] / merged["Memory (MB/img)_unet"]
merged["efficiency_ratio"] = merged["Efficiency (img/s/M)_fpn"] / merged["Efficiency (img/s/M)_unet"]

print("\nGlobal Statistics:")
print(f"  Average FPN/UNet throughput ratio: {merged['throughput_ratio'].mean():.2f}x")
print(f"  Average FPN/UNet memory ratio: {merged['memory_ratio'].mean():.2f}x")
print(f"  Average FPN/UNet efficiency ratio: {merged['efficiency_ratio'].mean():.2f}x")

print("\n" + "-"*80)
print("TOP 10 ENCODERS BY EFFICIENCY (FPN)")
print("-"*80)
top10_fpn = merged.nlargest(10, "Efficiency (img/s/M)_fpn")[
    ["Encoder", "Throughput (img/s)_fpn", "Memory (MB/img)_fpn", "Efficiency (img/s/M)_fpn", "Total Params (M)_fpn"]
]
top10_fpn.columns = ["Encoder", "Throughput", "Memory", "Efficiency", "Params (M)"]
print(top10_fpn.to_string(index=False))

print("\n" + "-"*80)
print("TOP 10 ENCODERS BY EFFICIENCY (UNet)")
print("-"*80)
top10_unet = merged.nlargest(10, "Efficiency (img/s/M)_unet")[
    ["Encoder", "Throughput (img/s)_unet", "Memory (MB/img)_unet", "Efficiency (img/s/M)_unet", "Total Params (M)_unet"]
]
top10_unet.columns = ["Encoder", "Throughput", "Memory", "Efficiency", "Params (M)"]
print(top10_unet.to_string(index=False))

print("\n" + "-"*80)
print("PARETO-OPTIMAL ENCODERS (FPN) - Not dominated on Throughput vs Memory")
print("-"*80)
print(", ".join(pareto_fpn))

# Best picks recommendation
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
Based on the analysis:

1. FPN decoder is generally FASTER (higher throughput) than UNet
2. FPN decoder uses LESS MEMORY than UNet
3. FPN decoder has BETTER EFFICIENCY (throughput per parameter)

Top picks by use case:
- Maximum speed (small model): mobileone_s0, mobilenetv3_small_100
- Good balance speed/accuracy: mobilenetv4_conv_small, mobileone_s1
- Higher capacity (larger model): mobilenetv4_conv_medium, mobileone_s2
""")

# Save comparison data
merged.to_csv("decoder_comparison.csv", index=False)
print("\nSaved: decoder_comparison.png, decoder_comparison.csv")
