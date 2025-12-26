# Semantic FPN Decoder Benchmark

## Architecture

This benchmark uses a standard **Semantic FPN** implementation, based on the architecture described in "Panoptic Feature Pyramid Networks" (Kirillov et al., CVPR 2019).

### Decoder Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `fpn_channels` | 128 | Balance between capacity and CPU speed |
| `out_indices` | (1, 2, 3, 4) | C2, C3, C4, C5 (strides 4, 8, 16, 32) |
| `output_level` | P2 only | Highest resolution, standard for segmentation |

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

### Differences with Previous Implementation

| Aspect | Previous (slow) | New (standard) |
|--------|-----------------|----------------|
| Levels used | All (P2-P5 concat) | P2 only |
| Final channels | 512 (128×4) | 128 |
| Upsample | 4× on all branches | 1× on P2 + final 4× |

## References

1. **Feature Pyramid Networks for Object Detection**
   - Lin et al., CVPR 2017
   - https://arxiv.org/abs/1612.03144
   - Original FPN architecture

2. **Panoptic Feature Pyramid Networks**
   - Kirillov et al., CVPR 2019
   - https://arxiv.org/abs/1901.02446
   - Semantic FPN for segmentation (uses P2)

3. **Detectron2 Implementation**
   - https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/semantic_seg.py
   - Reference implementation

## Optimizations

- `torch.compile()`: Full model compilation
- `torch.autocast(device_type="cpu")`: CPU mixed precision (bfloat16)
- `torch.inference_mode()`: Optimized inference mode
- Reparameterization for RepVGG/MobileOne

## Benchmark

```bash
cd benchmarks/fpn_decoder
uv run benchmark.py
```
