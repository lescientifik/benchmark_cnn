"""Compte le nombre de paramètres du modèle TotalSegmentator (nnU-Net 3d_fullres)."""

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp


def count_parameters(model):
    """Compte les paramètres totaux et entraînables."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    """Formate le nombre de paramètres."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def create_totalsegmentator_model():
    """
    Crée l'architecture nnU-Net 3d_fullres utilisée par TotalSegmentator.
    Configuration basée sur les plans de TotalSegmentator (task 291).
    """
    # Configuration standard nnU-Net 3d_fullres pour TotalSegmentator
    # Source: https://github.com/wasserth/TotalSegmentator et nnU-Net plans
    model = PlainConvUNet(
        input_channels=1,  # CT scan mono-canal
        n_stages=6,        # 6 stages de résolution
        features_per_stage=[32, 64, 128, 256, 320, 320],  # Nombre de features par stage
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6,  # Kernels 3x3x3
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],  # 2 convolutions par stage (encoder)
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],  # 2 convolutions par stage (decoder)
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
        nonlin_first=False,
        num_classes=117,  # TotalSegmentator v2: 117 classes
    )
    return model


if __name__ == "__main__":
    print("Création de l'architecture nnU-Net 3d_fullres (TotalSegmentator)...")
    model = create_totalsegmentator_model()

    total, trainable = count_parameters(model)
    print(f"\nNombre de paramètres:")
    print(f"  Total:        {total:>15,} ({format_params(total)})")
    print(f"  Entraînables: {trainable:>15,} ({format_params(trainable)})")
