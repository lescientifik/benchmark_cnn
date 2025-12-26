# Benchmark CNN - Segmentation Encoders & Decoders

Benchmark complet des architectures CNN pour la segmentation d'images sur CPU.
Comparaison des encodeurs (timm) avec décodeurs FPN et UNet, ainsi que des modèles 3D officiels (STU-Net, TotalSegmentator).

## Objectifs

1. **Évaluer les performances CPU** des différentes architectures d'encodeurs pour la segmentation
2. **Comparer les décodeurs FPN vs UNet** en termes de throughput et paramètres
3. **Mesurer l'impact des optimisations** (torch.compile, autocast, inference_mode)
4. **Comparer avec les modèles 3D de référence** (STU-Net, TotalSegmentator/nnU-Net)
5. **Identifier les meilleurs compromis** vitesse/paramètres pour le déploiement CPU

## Structure du projet

```
benchmark_cnn/
├── README.md                    # Ce fichier
├── CLAUDE.md                    # Instructions pour Claude Code
├── notes.md                     # Notes sur les modèles et résultats
├── plan.md                      # Plan d'exécution détaillé
├── pyproject.toml               # Configuration du projet (uv)
├── benchmark_official.py        # Benchmark 2D vs 3D (STU-Net, TotalSegmentator)
├── count_params.py              # Utilitaire pour compter les paramètres
└── benchmarks/
    ├── encoder_only/
    │   ├── benchmark.py         # Benchmark des encodeurs seuls
    │   ├── benchmark_results.csv
    │   ├── benchmark_results_all.csv
    │   ├── benchmark_results_comparison.csv
    │   └── benchmark_plot.png
    ├── fpn_decoder/
    │   ├── benchmark.py         # Benchmark encoder + Semantic FPN
    │   ├── README.md
    │   ├── benchmark_results.csv
    │   └── benchmark_plot.png
    ├── unet_decoder/
    │   ├── benchmark.py         # Benchmark encoder + UNet decoder
    │   ├── README.md
    │   ├── benchmark_results.csv
    │   └── benchmark_plot.png
    ├── combined_benchmark.py    # Comparaison FPN vs UNet
    ├── combined_benchmark_results.csv
    ├── combined_benchmark_plot.png
    └── batch_size_comparison.py # Impact du batch size
```

## Benchmarks

### 1. Encoder-Only (`benchmarks/encoder_only/`)

Benchmark des encodeurs de classification (sans décodeur) avec 4 modes d'optimisation:
- **baseline**: Sans optimisation
- **autocast**: torch.autocast(cpu, bfloat16)
- **compile**: torch.compile()
- **autocast+compile**: Les deux optimisations combinées

**Configuration**: Images 512x512, batch size 1, 50 itérations

**Métriques**:
- Throughput (img/s)
- Paramètres (M)
- Mémoire (MB/img)
- Speedup (autocast+compile vs baseline)
- Efficacité (throughput / params)

### 2. FPN Decoder (`benchmarks/fpn_decoder/`)

Benchmark encoder + Semantic FPN (Feature Pyramid Network):
- Architecture standard FPN (Panoptic FPN, Kirillov et al. 2019)
- Channels FPN: 128
- Utilise seulement P2 (plus haute résolution) pour la segmentation
- Optimisations: compile + autocast + inference_mode

**Overhead FPN**: ~0.8-1.0M paramètres supplémentaires

### 3. UNet Decoder (`benchmarks/unet_decoder/`)

Benchmark encoder + décodeur UNet classique:
- Skip connections par concaténation
- ConvTranspose2d pour l'upsampling
- Double conv blocks (Conv-BN-ReLU × 2)
- Channels adaptés à l'encodeur

**Overhead UNet**: ~2.5-3.5M paramètres supplémentaires

### 4. Modèles 3D Officiels (`benchmark_official.py`)

Comparaison avec les implémentations officielles pour la segmentation médicale 117 classes:

**STU-Net** (https://github.com/uni-medical/STU-Net):
- STU-Net-S: 14.6M params
- STU-Net-B: 58.3M params
- Architecture: BasicResBlock, 6 stages

**TotalSegmentator** (nnU-Net PlainConvUNet):
- 31.3M params
- Config: 6 stages, features [32, 64, 128, 256, 320, 320]
- InstanceNorm3d, LeakyReLU

## Familles de modèles testées

| Famille | Modèles | Plage params |
|---------|---------|--------------|
| MobileOne | s0, s1, s2, s3 | 2M - 10M |
| MobileNetV4 | small, medium, large | 4M - 33M |
| MobileNetV3 | small, large | 2.5M - 5.5M |
| RegNetX | 002, 004, 008, 016, 032 | 2.7M - 15M |
| RegNetY | 002, 004, 008, 016 | 3M - 11M |
| ResNet | 18, 34, 50 | 12M - 26M |
| EfficientNet | b0, b1, b2 | 5M - 9M |
| RepVGG | a0, a1, a2, b0 | 8M - 25M |
| LCNet | 100, 150 | 3M - 4.5M |
| TinyNet | a, b, c | 2.5M - 6M |
| EdgeNeXt | x_small, small | 2M - 6M |
| ConvNeXt | tiny, small | 29M - 50M |
| GhostNetV2 | 100, 130, 160 | 6M - 12M |
| DenseNet | 121, 169 | 8M - 14M |
| ViT | tiny, small | 6M - 22M |
| PoolFormer | s12, s24 | 12M - 21M |

## Résultats clés

### Encodeurs seuls (512x512, CPU)

| Modèle | Params | Throughput | Efficacité |
|--------|--------|------------|------------|
| mobileone_s0 | 2.1M | 241 img/s | 115.9 |
| regnetx_002 | 2.7M | 215 img/s | 80.1 |
| mobilenetv4_small | 3.8M | 236 img/s | 62.5 |
| lcnet_100 | 3.0M | 202 img/s | 68.4 |
| mobilenetv3_small | 2.5M | 168 img/s | 65.9 |

### FPN vs UNet

- **FPN**: Plus léger (~1M params), plus rapide (1.3-1.5x speedup)
- **UNet**: Plus de capacité (~3M params), meilleur pour segmentation fine

### 2D vs 3D (segmentation 117 classes)

| Modèle | Type | Params | Throughput equiv |
|--------|------|--------|------------------|
| mobileone_s0 + UNet | 2D | 4.6M | 27 img/s |
| STU-Net-S | 3D | 14.6M | 12.9 eq/s |
| TotalSegmentator | 3D | 31.3M | 6.8 eq/s |

**Conclusion**: Les modèles 2D sont ~2x plus rapides que les 3D pour un nombre de paramètres similaire.

## Installation

```bash
# Cloner le repo
git clone <repo-url>
cd benchmark_cnn

# Installer les dépendances avec uv
uv sync
```

## Exécution

```bash
# Benchmark encodeurs seuls
cd benchmarks/encoder_only
uv run benchmark.py

# Benchmark FPN decoder
cd benchmarks/fpn_decoder
uv run benchmark.py

# Benchmark UNet decoder
cd benchmarks/unet_decoder
uv run benchmark.py

# Benchmark 2D vs 3D officiel
uv run benchmark_official.py

# Comparaison FPN vs UNet
cd benchmarks
uv run combined_benchmark.py
```

## Dépendances principales

- `torch >= 2.9.1` - PyTorch
- `torchvision >= 0.24.1` - Computer vision
- `timm >= 1.0.22` - PyTorch Image Models (encodeurs)
- `dynamic_network_architectures` - nnU-Net architectures (TotalSegmentator)
- `totalsegmentator >= 2.12.0` - Pour référence
- `matplotlib`, `pandas`, `psutil`, `tabulate`, `tqdm`

## Optimisations utilisées

1. **torch.compile()** - Compilation JIT pour CPU
2. **torch.autocast(cpu, bfloat16)** - Mixed precision
3. **torch.inference_mode()** - Mode inférence optimisé
4. **Reparameterization** - Pour RepVGG et MobileOne
5. **model.eval()** - Mode évaluation

## Références

- [STU-Net](https://arxiv.org/abs/2304.06716) - Scalable and Transferable U-Net
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - 117 anatomical structures
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - Self-configuring segmentation
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Panoptic FPN](https://arxiv.org/abs/1901.02446) - Semantic FPN architecture
