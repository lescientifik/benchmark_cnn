#!/usr/bin/env python3
"""Comparison plot: 2D models (FPN/UNet) vs 3D models (STU-Net/TotalSegmentator).

This script creates a comparison chart showing the performance gains
of 2D segmentation models over 3D models at similar parameter counts.
"""

import gc
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Import nnU-Net PlainConvUNet (used by TotalSegmentator)
from dynamic_network_architectures.architectures.unet import PlainConvUNet

NUM_CLASSES = 117


# =============================================================================
# STU-Net - Official code from https://github.com/uni-medical/STU-Net
# =============================================================================

class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)]
        )
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool+1):
            stage = nn.Sequential(
                BasicResBlock(dims[d-1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)]
            )
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1-u], dims[-2-u], pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(dims[-2-u] * 2, dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u], use_1x1conv=True),
                *[BasicResBlock(dims[-2-u], dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u]) for _ in range(depth[-2-u]-1)]
            )
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


# =============================================================================
# 2D Models
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDecoder2D(nn.Module):
    def __init__(self, encoder_channels, num_classes, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        self.blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        in_ch = encoder_channels[0]
        for enc_ch, dec_ch in zip(encoder_channels[1:], decoder_channels):
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


class SemanticFPNDecoder(nn.Module):
    """Semantic FPN decoder."""

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
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="nearest"
            )

        fpn_features = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]
        p2 = fpn_features[0]

        return self.seg_head(p2)


class SegModel2D(nn.Module):
    def __init__(self, encoder_name, num_classes=NUM_CLASSES, decoder_type="unet"):
        super().__init__()

        if decoder_type == "unet":
            self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3, 4))
        else:  # FPN
            self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.encoder(dummy)
        encoder_channels = [f.shape[1] for f in features]

        if decoder_type == "unet":
            max_ch = max(encoder_channels)
            decoder_channels = [min(256, max_ch), min(128, max_ch // 2), min(64, max_ch // 4), min(32, max_ch // 8)]
            self.decoder = UNetDecoder2D(encoder_channels, num_classes, decoder_channels)
        else:
            self.decoder = SemanticFPNDecoder(encoder_channels, num_classes, fpn_channels=128)

        self.decoder_type = decoder_type

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# =============================================================================
# Model creation functions
# =============================================================================

def create_stunet_s():
    """STU-Net Small - official config."""
    return STUNet(
        input_channels=1,
        num_classes=NUM_CLASSES,
        depth=[1]*6,
        dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
        pool_op_kernel_sizes=[[2,2,2]]*5,
        conv_kernel_sizes=[[3,3,3]]*6,
        enable_deep_supervision=False,
    )


def create_stunet_b():
    """STU-Net Base - official config."""
    return STUNet(
        input_channels=1,
        num_classes=NUM_CLASSES,
        depth=[1]*6,
        dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
        pool_op_kernel_sizes=[[2,2,2]]*5,
        conv_kernel_sizes=[[3,3,3]]*6,
        enable_deep_supervision=False,
    )


def create_totalsegmentator():
    """TotalSegmentator = nnU-Net 3d_fullres PlainConvUNet."""
    return PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
        nonlin_first=False,
        num_classes=NUM_CLASSES,
    )


# =============================================================================
# Benchmark function
# =============================================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark(model, input_tensor, warmup=3, iterations=20):
    model.eval()
    try:
        model = torch.compile(model)
    except Exception:
        pass

    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(warmup):
            _ = model(input_tensor)

    times = []
    with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            times.append(time.perf_counter() - start)

    return sum(times) / len(times)


# =============================================================================
# Main comparison
# =============================================================================

if __name__ == "__main__":
    print("=" * 85)
    print(f"COMPARISON: 2D Models vs 3D Models for {NUM_CLASSES}-class Segmentation")
    print("=" * 85)

    results = []

    # 3D models
    print("\n### 3D Models @ 128x128x64 ###")
    input_3d = torch.randn(1, 1, 128, 128, 64)
    equiv_slices = (128 * 128 * 64) / (512 * 512)

    models_3d = [
        ("STU-Net-S", create_stunet_s),
        ("TotalSegmentator", create_totalsegmentator),
    ]

    for name, create_fn in models_3d:
        try:
            model = create_fn()
            params = count_params(model)
            avg_time = benchmark(model, input_3d)
            throughput = 1.0 / avg_time
            equiv_tp = equiv_slices / avg_time
            results.append({
                "Model": name,
                "Type": "3D",
                "Params (M)": params / 1e6,
                "Time (ms)": avg_time * 1000,
                "Throughput (vol/s)": throughput,
                "Equiv Throughput (img/s)": equiv_tp,
            })
            print(f"  {name}: {params/1e6:.2f}M params, {avg_time*1000:.1f}ms, {equiv_tp:.1f} equiv/s")
            del model
            gc.collect()
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # 2D models at similar param counts
    print("\n### 2D Models @ 512x512 (similar params to 3D) ###")
    input_2d = torch.randn(1, 3, 512, 512)

    # Models chosen to match ~15M (STU-Net-S) and ~31M (TotalSegmentator)
    models_2d = [
        # Small models (< 15M params)
        ("mnv4_small + UNet", "mobilenetv4_conv_small.e2400_r224_in1k", "unet"),
        ("mnv4_small + FPN", "mobilenetv4_conv_small.e2400_r224_in1k", "fpn"),
        ("mnv4_medium + UNet", "mobilenetv4_conv_medium.e500_r224_in1k", "unet"),
        ("mnv4_medium + FPN", "mobilenetv4_conv_medium.e500_r224_in1k", "fpn"),
        # Around 15M params (STU-Net-S range)
        ("resnet18 + UNet", "resnet18", "unet"),
        ("resnet18 + FPN", "resnet18", "fpn"),
        ("regnetx_016 + UNet", "regnetx_016", "unet"),
        ("regnetx_016 + FPN", "regnetx_016", "fpn"),
        ("mobileone_s3 + UNet", "mobileone_s3", "unet"),
        ("mobileone_s3 + FPN", "mobileone_s3", "fpn"),
        # Around 31M params (TotalSegmentator range)
        ("mnv4_large + UNet", "mobilenetv4_conv_large.e600_r384_in1k", "unet"),
        ("mnv4_large + FPN", "mobilenetv4_conv_large.e600_r384_in1k", "fpn"),
        ("resnet34 + UNet", "resnet34", "unet"),
        ("resnet34 + FPN", "resnet34", "fpn"),
        ("resnet50 + UNet", "resnet50", "unet"),
        ("resnet50 + FPN", "resnet50", "fpn"),
    ]

    for name, encoder, decoder_type in models_2d:
        try:
            model = SegModel2D(encoder, NUM_CLASSES, decoder_type)
            if any(x in encoder for x in ["repvgg", "mobileone", "fastvit"]):
                for m in model.modules():
                    if hasattr(m, "reparameterize"):
                        m.reparameterize()
            params = count_params(model)
            avg_time = benchmark(model, input_2d)
            throughput = 1.0 / avg_time
            results.append({
                "Model": name,
                "Type": "2D",
                "Params (M)": params / 1e6,
                "Time (ms)": avg_time * 1000,
                "Throughput (vol/s)": throughput,  # For 2D, vol/s = img/s
                "Equiv Throughput (img/s)": throughput,
            })
            print(f"  {name}: {params/1e6:.2f}M params, {avg_time*1000:.1f}ms, {throughput:.1f} img/s")
            del model
            gc.collect()
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_csv("comparison_2d_vs_3d.csv", index=False)
    print("\nResults saved to comparison_2d_vs_3d.csv")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Throughput comparison grouped by param range
    ax1 = axes[0]

    # Split by param range
    small_models = df[df["Params (M)"] < 20].sort_values("Equiv Throughput (img/s)", ascending=True)
    large_models = df[df["Params (M)"] >= 20].sort_values("Equiv Throughput (img/s)", ascending=True)

    colors_small = ["#e41a1c" if t == "3D" else "#377eb8" for t in small_models["Type"]]
    colors_large = ["#e41a1c" if t == "3D" else "#377eb8" for t in large_models["Type"]]

    # Small models subplot
    y_pos = range(len(small_models))
    ax1.barh(y_pos, small_models["Equiv Throughput (img/s)"], color=colors_small, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{m} ({p:.1f}M)" for m, p in zip(small_models["Model"], small_models["Params (M)"])], fontsize=9)
    ax1.set_xlabel("Throughput (equiv img/s)", fontsize=12)
    ax1.set_title("Models < 20M params\n(comparable to STU-Net-S)", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="x")

    # Add 3D reference lines
    stunet_s = df[df["Model"] == "STU-Net-S"]["Equiv Throughput (img/s)"].values
    if len(stunet_s) > 0:
        ax1.axvline(x=stunet_s[0], color="red", linestyle="--", alpha=0.7, label=f"STU-Net-S: {stunet_s[0]:.1f}")
        ax1.legend(loc="lower right")

    # Large models subplot
    ax2 = axes[1]
    y_pos = range(len(large_models))
    ax2.barh(y_pos, large_models["Equiv Throughput (img/s)"], color=colors_large, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{m} ({p:.1f}M)" for m, p in zip(large_models["Model"], large_models["Params (M)"])], fontsize=9)
    ax2.set_xlabel("Throughput (equiv img/s)", fontsize=12)
    ax2.set_title("Models >= 20M params\n(comparable to TotalSegmentator)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="x")

    # Add 3D reference lines
    ts = df[df["Model"] == "TotalSegmentator"]["Equiv Throughput (img/s)"].values
    if len(ts) > 0:
        ax2.axvline(x=ts[0], color="red", linestyle="--", alpha=0.7, label=f"TotalSegmentator: {ts[0]:.1f}")
        ax2.legend(loc="lower right")

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e41a1c', label='3D Model'),
                       Patch(facecolor='#377eb8', label='2D Model')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("2D vs 3D Segmentation Models: Throughput Comparison\n(CPU benchmark, 117 classes)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("comparison_2d_vs_3d.png", dpi=150, bbox_inches="tight")
    print("Plot saved to comparison_2d_vs_3d.png")

    # Print speedup summary
    print("\n" + "=" * 85)
    print("SPEEDUP SUMMARY: 2D vs 3D at Similar Parameter Counts")
    print("=" * 85)

    # STU-Net-S comparison
    stunet_s_tp = df[df["Model"] == "STU-Net-S"]["Equiv Throughput (img/s)"].values
    if len(stunet_s_tp) > 0:
        stunet_s_tp = stunet_s_tp[0]
        print(f"\nSTU-Net-S baseline: {stunet_s_tp:.1f} equiv img/s")
        small_2d = df[(df["Type"] == "2D") & (df["Params (M)"] < 20)]
        for _, row in small_2d.iterrows():
            speedup = row["Equiv Throughput (img/s)"] / stunet_s_tp
            print(f"  {row['Model']}: {row['Equiv Throughput (img/s)']:.1f} equiv img/s ({speedup:.1f}x speedup)")

    # TotalSegmentator comparison
    ts_tp = df[df["Model"] == "TotalSegmentator"]["Equiv Throughput (img/s)"].values
    if len(ts_tp) > 0:
        ts_tp = ts_tp[0]
        print(f"\nTotalSegmentator baseline: {ts_tp:.1f} equiv img/s")
        large_2d = df[(df["Type"] == "2D") & (df["Params (M)"] >= 20)]
        for _, row in large_2d.iterrows():
            speedup = row["Equiv Throughput (img/s)"] / ts_tp
            print(f"  {row['Model']}: {row['Equiv Throughput (img/s)']:.1f} equiv img/s ({speedup:.1f}x speedup)")
