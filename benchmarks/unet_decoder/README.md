# UNet Decoder Benchmark

## Architecture

Ce benchmark utilise une implémentation **UNet decoder** standard, basée sur l'architecture originale de Ronneberger et al. (2015), adaptée pour utiliser des encodeurs pré-entraînés de timm.

### Paramètres du décodeur

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `out_indices` | (0, 1, 2, 3, 4) | Tous les niveaux de features (C1-C5) |
| `decoder_channels` | (256, 128, 64, 32) | Réduction progressive, adapté à l'encoder |
| `upsampling` | ConvTranspose2d (stride=2) | Standard UNet |
| `conv_block` | Double conv 3x3 + BN + ReLU | Bloc UNet classique |

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

### Bloc de convolution (ConvBlock)

```python
Conv2d(in_ch, out_ch, 3, padding=1) → BN → ReLU
Conv2d(out_ch, out_ch, 3, padding=1) → BN → ReLU
```

### Différences avec FPN

| Aspect | UNet | Semantic FPN |
|--------|------|--------------|
| Skip connections | Concatenation | Addition |
| Traitement | Séquentiel (deep→shallow) | Parallèle puis fusion |
| Nb de niveaux utilisés | Tous (5) | Principalement P2 |
| Bloc après fusion | Double conv 3x3 | Simple conv 3x3 |

## Références

1. **U-Net: Convolutional Networks for Biomedical Image Segmentation**
   - Ronneberger et al., MICCAI 2015
   - https://arxiv.org/abs/1505.04597
   - Architecture originale

2. **Segmentation Models PyTorch**
   - https://github.com/qubvel/segmentation_models.pytorch
   - Implémentation de référence avec encodeurs timm

3. **nnU-Net**
   - Isensee et al., Nature Methods 2021
   - https://github.com/MIC-DKFZ/nnUNet
   - Version auto-configurée pour médical

## Optimisations

- `torch.compile()` : Compilation du modèle complet
- `torch.autocast(device_type="cpu")` : Mixed precision CPU (bfloat16)
- `torch.inference_mode()` : Mode inférence optimisé
- Reparamétrisation pour RepVGG/MobileOne

## Benchmark

```bash
cd benchmarks/unet_decoder
uv run benchmark.py
```
