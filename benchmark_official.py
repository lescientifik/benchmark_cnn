#!/usr/bin/env python3
"""Benchmark avec les implémentations OFFICIELLES de STU-Net et nnU-Net (TotalSegmentator).

STU-Net: https://github.com/uni-medical/STU-Net
TotalSegmentator: https://github.com/wasserth/TotalSegmentator (utilise nnU-Net PlainConvUNet)
"""

import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Importer nnU-Net PlainConvUNet (utilisé par TotalSegmentator)
from dynamic_network_architectures.architectures.unet import PlainConvUNet

NUM_CLASSES = 117


# =============================================================================
# STU-Net - Code officiel de https://github.com/uni-medical/STU-Net
# Fichier: nnUNet-2.2/nnunetv2/training/nnUNetTrainer/STUNetTrainer.py
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


# === Créer les modèles avec les configs officielles ===

def create_stunet_s():
    """STU-Net Small - config officielle du repo."""
    return STUNet(
        input_channels=1,
        num_classes=NUM_CLASSES,
        depth=[1]*6,
        dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],  # [16, 32, 64, 128, 256, 256]
        pool_op_kernel_sizes=[[2,2,2]]*5,
        conv_kernel_sizes=[[3,3,3]]*6,
        enable_deep_supervision=False,
    )


def create_stunet_b():
    """STU-Net Base - config officielle du repo."""
    return STUNet(
        input_channels=1,
        num_classes=NUM_CLASSES,
        depth=[1]*6,
        dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],  # [32, 64, 128, 256, 512, 512]
        pool_op_kernel_sizes=[[2,2,2]]*5,
        conv_kernel_sizes=[[3,3,3]]*6,
        enable_deep_supervision=False,
    )


def create_totalsegmentator():
    """TotalSegmentator = nnU-Net 3d_fullres PlainConvUNet avec config standard."""
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


# === Modèles 2D ===

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


class SegModel2D(nn.Module):
    def __init__(self, encoder_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3, 4))

        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.encoder(dummy)
        encoder_channels = [f.shape[1] for f in features]

        max_ch = max(encoder_channels)
        decoder_channels = [min(256, max_ch), min(128, max_ch // 2), min(64, max_ch // 4), min(32, max_ch // 8)]
        self.decoder = UNetDecoder2D(encoder_channels, num_classes, decoder_channels)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# === Benchmark ===

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def format_params(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.1f}K"


def benchmark(model, input_tensor, warmup=3, iterations=10):
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


ENCODERS_2D = ["mobileone_s0", "regnetx_002", "mobilenetv3_small_100", "resnet18"]


if __name__ == "__main__":
    print("=" * 85)
    print(f"BENCHMARK SEGMENTATION {NUM_CLASSES} CLASSES - Implémentations OFFICIELLES")
    print("=" * 85)

    # === 2D ===
    print("\n### Modèles 2D (timm encoder + UNet decoder) @ 512×512 ###")
    input_2d = torch.randn(1, 3, 512, 512)
    print(f"{'Modèle':<35} {'Params':>12} {'Temps':>10} {'img/s':>10} {'Eff':>10}")
    print("-" * 77)

    for enc in ENCODERS_2D:
        try:
            model = SegModel2D(enc)
            if "mobileone" in enc:
                for m in model.modules():
                    if hasattr(m, "reparameterize"):
                        m.reparameterize()
            params = count_params(model)
            avg_time = benchmark(model, input_2d)
            throughput = 1.0 / avg_time
            eff = throughput / (params / 1e6)
            print(f"{enc + ' + UNet':<35} {format_params(params):>12} {avg_time*1000:>8.1f}ms {throughput:>8.1f}/s {eff:>10.2f}")
            del model
            gc.collect()
        except Exception as e:
            print(f"{enc:<35} ERREUR: {e}")

    # === 3D ===
    print("\n### Modèles 3D (STU-Net / TotalSegmentator) @ 128×128×64 ###")
    input_3d = torch.randn(1, 1, 128, 128, 64)
    equiv_slices = (128 * 128 * 64) / (512 * 512)

    print(f"Volume = {equiv_slices:.1f} équivalent slices 512×512")
    print(f"{'Modèle':<35} {'Params':>12} {'Temps':>10} {'eq/s':>10} {'Eff':>10}")
    print("-" * 77)

    models_3d = [
        ("STU-Net-S (officiel)", create_stunet_s),
        ("STU-Net-B (officiel)", create_stunet_b),
        ("TotalSegmentator (PlainConvUNet)", create_totalsegmentator),
    ]

    for name, create_fn in models_3d:
        try:
            model = create_fn()
            params = count_params(model)
            avg_time = benchmark(model, input_3d)
            equiv_tp = equiv_slices / avg_time
            eff = equiv_tp / (params / 1e6)
            print(f"{name:<35} {format_params(params):>12} {avg_time*1000:>8.1f}ms {equiv_tp:>8.1f}/s {eff:>10.2f}")
            del model
            gc.collect()
        except Exception as e:
            print(f"{name:<35} ERREUR: {e}")

    print("\n" + "=" * 85)
    print("Note: eq/s = throughput équivalent en images 512×512")
    print("      Eff = throughput / params(M)")
    print("      STU-Net: https://github.com/uni-medical/STU-Net")
    print("      TotalSegmentator: PlainConvUNet de dynamic_network_architectures (nnU-Net)")
