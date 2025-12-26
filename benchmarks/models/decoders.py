"""Decoder architectures for segmentation models."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFPNDecoder(nn.Module):
    """Semantic FPN decoder - standard implementation.

    Architecture:
    - Lateral connections: 1x1 conv to reduce channels to fpn_channels
    - Top-down pathway: upsample + add
    - Uses only P2 (highest resolution) for final prediction

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
    """Encoder + decoder for segmentation.

    Args:
        encoder_name: Name of timm encoder
        decoder_type: "fpn" or "unet"
        num_classes: Number of output classes
        fpn_channels: Number of channels in FPN (only for FPN decoder)
    """

    def __init__(self, encoder_name, decoder_type="fpn", num_classes=1, fpn_channels=128):
        super().__init__()
        self.input_size = 512
        self.decoder_type = decoder_type

        # Create encoder with feature extraction
        if decoder_type == "fpn":
            out_indices = (1, 2, 3, 4)  # C2, C3, C4, C5
        else:  # unet
            out_indices = (0, 1, 2, 3, 4)  # All features

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=False,
            features_only=True,
            out_indices=out_indices,
        )

        # Get feature channels from encoder
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
        with torch.no_grad():
            features = self.encoder(dummy_input)
        encoder_channels = [f.shape[1] for f in features]

        # Create decoder
        if decoder_type == "fpn":
            self.decoder = SemanticFPNDecoder(encoder_channels, num_classes, fpn_channels)
        else:  # unet
            max_ch = max(encoder_channels)
            decoder_channels = [
                min(256, max_ch),
                min(128, max_ch // 2),
                min(64, max_ch // 4),
                min(32, max_ch // 8),
            ]
            self.decoder = UNetDecoder(encoder_channels, num_classes, decoder_channels)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)

        # Final upsample to input resolution
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out
