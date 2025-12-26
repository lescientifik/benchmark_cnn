# UNet Decoder Benchmark

## Architecture

This benchmark uses a standard **UNet decoder** implementation, based on the original architecture by Ronneberger et al. (2015), adapted to use pre-trained encoders from timm.

### Decoder Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `out_indices` | (0, 1, 2, 3, 4) | All feature levels (C1-C5) |
| `decoder_channels` | (256, 128, 64, 32) | Progressive reduction, adapted to encoder |
| `upsampling` | ConvTranspose2d (stride=2) | Standard UNet |
| `conv_block` | Double conv 3x3 + BN + ReLU | Classic UNet block |

### Pipeline

```
Encoder → [C1, C2, C3, C4, C5]
                          ↓
                    C5 (bottleneck)
                          ↓
           ConvTranspose2d ↑ + concat(C4) → ConvBlock
                          ↓
           ConvTranspose2d ↑ + concat(C3) → ConvBlock
                          ↓
           ConvTranspose2d ↑ + concat(C2) → ConvBlock
                          ↓
           ConvTranspose2d ↑ + concat(C1) → ConvBlock
                          ↓
           Final upsample → Seg head → Output
```

### Convolution Block (ConvBlock)

```python
Conv2d(in_ch, out_ch, 3, padding=1) → BN → ReLU
Conv2d(out_ch, out_ch, 3, padding=1) → BN → ReLU
```

### Differences with FPN

| Aspect | UNet | Semantic FPN |
|--------|------|--------------|
| Skip connections | Concatenation | Addition |
| Processing | Sequential (deep→shallow) | Parallel then fusion |
| Levels used | All (5) | Mainly P2 |
| Block after fusion | Double conv 3x3 | Simple conv 3x3 |

## References

1. **U-Net: Convolutional Networks for Biomedical Image Segmentation**
   - Ronneberger et al., MICCAI 2015
   - https://arxiv.org/abs/1505.04597
   - Original architecture

2. **Segmentation Models PyTorch**
   - https://github.com/qubvel/segmentation_models.pytorch
   - Reference implementation with timm encoders

3. **nnU-Net**
   - Isensee et al., Nature Methods 2021
   - https://github.com/MIC-DKFZ/nnUNet
   - Auto-configured version for medical imaging

## Optimizations

- `torch.compile()`: Full model compilation
- `torch.autocast(device_type="cpu")`: CPU mixed precision (bfloat16)
- `torch.inference_mode()`: Optimized inference mode
- Reparameterization for RepVGG/MobileOne

## Benchmark

```bash
cd benchmarks/unet_decoder
uv run benchmark.py
```
