# Notes sur les modèles de segmentation

## TotalSegmentator

TotalSegmentator utilise nnU-Net comme architecture de base.

### Taille du modèle
- **nnU-Net 3d_fullres (TotalSegmentator)** : **31,288,169 paramètres** (~31.29M)
- Vérifié via PyTorch avec l'architecture PlainConvUNet (config: 6 stages, features [32,64,128,256,320,320], 117 classes)
- Source originale: [GitHub Discussion #516](https://github.com/MIC-DKFZ/nnUNet/discussions/516)

### Variantes STU-Net (basées sur nnU-Net + TotalSegmentator)
| Modèle | Paramètres | FLOPs | DSC (%) |
|--------|------------|-------|---------|
| nnU-Net (baseline) | 31.28M | - | 86.76 |
| STU-Net-S | 14.6M | 0.13T | 83.74 |
| STU-Net-B | 58.26M | 0.51T | 87.12 |
| STU-Net-L | 440.30M | 3.81T | 88.71 |
| STU-Net-H | **1457.33M** (~1.46B) | 12.60T | **90.06** |

- STU-Net-H est le plus grand modèle de segmentation médicale pré-entraîné
- STU-Net-B dépasse nnU-Net baseline avec ~2x les paramètres
- Source: [STU-Net arXiv:2304.06716](https://arxiv.org/abs/2304.06716), [GitHub](https://github.com/openmedlab/STU-Net)

## Benchmark segmentation 117 classes CPU

Comparaison modèles 2D (timm + UNet) vs 3D (STU-Net/TotalSegmentator).
**Implémentations officielles** : STU-Net du repo GitHub, TotalSegmentator via PlainConvUNet (nnU-Net).

### Modèles 2D @ 512×512

| Modèle | Params | Temps | img/s | Efficacité |
|--------|--------|-------|-------|------------|
| mobilenetv3_small + UNet | 3.42M | 33ms | 30.2/s | 8.84 |
| mobileone_s0 + UNet | 4.63M | 37ms | 27.2/s | 5.87 |
| regnetx_002 + UNet | 4.88M | 37ms | 27.3/s | 5.59 |
| resnet18 + UNet | 14.24M | 52ms | 19.3/s | 1.36 |

### Modèles 3D @ 128×128×64 (= 4 slices 512×512 équivalentes)

| Modèle | Params | Temps | equiv/s | Efficacité |
|--------|--------|-------|---------|------------|
| STU-Net-S | 14.60M | 311ms | 12.9/s | 0.88 |
| TotalSegmentator | 31.29M | 591ms | 6.8/s | 0.22 |
| STU-Net-B | 58.27M | 804ms | 5.0/s | 0.09 |

**Conclusions**:
- Les modèles 2D sont **~2x plus rapides** que STU-Net-S pour une taille similaire
- TotalSegmentator (31M) est **2x plus lent** que STU-Net-S (15M) mais plus précis (DSC 86.76% vs 83.74%)
- STU-Net-S offre le meilleur compromis vitesse/précision pour la segmentation 3D

## Benchmark encodeurs CPU @ 512x512

Avec `torch.compile` + `autocast` + `inference_mode`:

| Modèle | Params | Throughput | Efficacité |
|--------|--------|------------|------------|
| mobileone_s0 | 2.1M | 241 img/s | 115.9 |
| regnetx_002 | 2.7M | 215 img/s | 80.1 |
| lcnet_100 | 3.0M | 202 img/s | 68.4 |
| mobilenetv4_conv_small | 3.8M | 236 img/s | 62.5 |
| mobilenetv3_small | 2.5M | 168 img/s | 65.9 |
