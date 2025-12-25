# Semantic FPN Decoder Benchmark

## Architecture

Ce benchmark utilise une implémentation **Semantic FPN** standard, basée sur l'architecture décrite dans "Panoptic Feature Pyramid Networks" (Kirillov et al., CVPR 2019).

### Paramètres du décodeur

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `fpn_channels` | 128 | Compromis entre capacité et vitesse CPU |
| `out_indices` | (1, 2, 3, 4) | C2, C3, C4, C5 (strides 4, 8, 16, 32) |
| `output_level` | P2 uniquement | Plus haute résolution, standard pour segmentation |

### Pipeline

```
Encoder → [C2, C3, C4, C5]
              ↓
Lateral convs (1x1) → reduce to fpn_channels
              ↓
Top-down pathway (upsample + add)
              ↓
Smooth convs (3x3) → [P2, P3, P4, P5]
              ↓
P2 → Seg head → Bilinear upsample → Output
```

### Différences avec mon ancienne implémentation

| Aspect | Ancienne (lente) | Nouvelle (standard) |
|--------|------------------|---------------------|
| Niveaux utilisés | Tous (P2-P5 concat) | P2 uniquement |
| Channels finaux | 512 (128×4) | 128 |
| Upsample | 4× sur toutes les branches | 1× sur P2 + final 4× |

## Références

1. **Feature Pyramid Networks for Object Detection**
   - Lin et al., CVPR 2017
   - https://arxiv.org/abs/1612.03144
   - Architecture FPN originale

2. **Panoptic Feature Pyramid Networks**
   - Kirillov et al., CVPR 2019
   - https://arxiv.org/abs/1901.02446
   - Semantic FPN pour segmentation (utilise P2)

3. **Detectron2 Implementation**
   - https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/semantic_seg.py
   - Implémentation de référence

## Optimisations

- `torch.compile()` : Compilation du modèle complet
- `torch.autocast(device_type="cpu")` : Mixed precision CPU (bfloat16)
- `torch.inference_mode()` : Mode inférence optimisé
- Reparamétrisation pour RepVGG/MobileOne

## Benchmark

```bash
cd benchmarks/fpn_decoder
uv run benchmark.py
```
