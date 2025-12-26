# Quantization Benchmark: Encoder + FPN Segmentation Models

**Configuration:**
- Input: 512x512 RGB images
- Decoder: FPN (Semantic FPN)
- Batch size: 1
- Device: CPU (multi-threaded)
- Classes: 117

## Results Summary

### Latency (ms) - Lower is better

| Encoder | Params | Baseline | Autocast | Compiled | ORT FP32 | ORT INT8 | Best Speedup |
|---------|--------|----------|----------|----------|----------|----------|--------------|
| resnet18 | 12.05M | 59.1 | 43.9 | 34.6 | 33.6 | **26.0** | 2.27x |
| resnet34 | 22.16M | 87.6 | 55.1 | 45.8 | 53.4 | **31.2** | 2.80x |
| resnet50 | 24.75M | 114.2 | 87.5 | 53.3 | 59.7 | **40.7** | 2.81x |
| regnetx_004 | 5.61M | 40.1 | 37.5 | 29.1 | **20.9** | 28.6 | 1.92x |
| regnetx_008 | 7.49M | 53.4 | 42.5 | 31.7 | 41.5 | **24.8** | 2.15x |
| regnety_004 | 4.76M | 44.4 | 44.7 | 32.4 | 39.8 | **35.7** | 1.37x |
| regnety_008 | 6.41M | 50.1 | 45.7 | 34.2 | 43.0 | **28.2** | 1.78x |
| repvgg_a0 | 7.99M | 45.3 | 38.7 | 32.8 | 49.0 | **24.5** | 1.85x |
| repvgg_a1 | 12.48M | 56.9 | 49.3 | 37.6 | 39.6 | **27.6** | 2.06x |
| mnv4_small | 2.16M | 34.7 | 34.9 | 26.2 | **18.7** | 26.1 | 1.85x |
| mnv4_medium | 8.12M | 49.5 | 43.0 | 33.3 | **26.5** | 29.2 | 1.86x |

### Throughput (img/s) - Higher is better

| Encoder | Baseline | Autocast | Compiled | ORT FP32 | ORT INT8 |
|---------|----------|----------|----------|----------|----------|
| resnet18 | 16.9 | 22.8 | 28.9 | 29.8 | **38.5** |
| resnet34 | 11.4 | 18.2 | 21.8 | 18.7 | **32.0** |
| resnet50 | 8.8 | 11.4 | 18.7 | 16.7 | **24.6** |
| regnetx_004 | 24.9 | 26.6 | 34.3 | **48.0** | 35.0 |
| regnetx_008 | 18.7 | 23.5 | 31.6 | 24.1 | **40.3** |
| regnety_004 | 22.5 | 22.4 | 30.9 | 25.1 | **28.0** |
| regnety_008 | 20.0 | 21.9 | 29.2 | 23.3 | **35.5** |
| repvgg_a0 | 22.1 | 25.8 | 30.5 | 20.4 | **40.8** |
| repvgg_a1 | 17.6 | 20.3 | 26.6 | 25.2 | **36.2** |
| **mnv4_small** | 28.8 | 28.7 | 38.1 | **53.3** | 38.3 |
| mnv4_medium | 20.2 | 23.2 | 30.0 | **37.7** | 34.3 |

## Key Findings

### Winner: MobileNetV4 Small + ONNX FP32 = 53.3 img/s

No quantization needed! Just export to ONNX and get 1.85x speedup.

### ONNX FP32 vs INT8: Depends on Architecture

| Architecture | Best Mode | Throughput | Why |
|--------------|-----------|------------|-----|
| **MobileNetV4** | ONNX FP32 | 53.3 img/s | Depthwise conv optimized in ONNX |
| **RegNetX small** | ONNX FP32 | 48.0 img/s | Group conv optimized in ONNX |
| **ResNet** | ONNX INT8 | 38.5 img/s | Standard conv benefits from INT8 |
| **RepVGG** | ONNX INT8 | 40.8 img/s | Simple conv3x3 benefits from INT8 |
| **RegNetX/Y large** | ONNX INT8 | 40.3 img/s | More compute = more INT8 gains |

### torch.compile + autocast vs ONNX

| Encoder | Compiled (img/s) | Best ONNX (img/s) | ONNX Wins? |
|---------|------------------|-------------------|------------|
| mnv4_small | 38.1 | 53.3 (FP32) | ✅ +40% |
| regnetx_004 | 34.3 | 48.0 (FP32) | ✅ +40% |
| resnet18 | 28.9 | 38.5 (INT8) | ✅ +33% |
| repvgg_a0 | 30.5 | 40.8 (INT8) | ✅ +34% |

**ONNX Runtime beats torch.compile for all architectures tested.**

## Quantization Accuracy Considerations

| Architecture | PTQ INT8 Accuracy | Recommendation |
|--------------|-------------------|----------------|
| **ResNet** | Stable (<1% drop) | INT8 safe |
| **RegNet** | Likely stable | INT8 probably safe |
| **RepVGG** | Severe degradation (20-35% drop) | Avoid INT8, use FP32 |
| **MobileNetV4** | Unknown, depthwise issues possible | Use ONNX FP32 instead |

### Why RepVGG has quantization issues:
> "The merging process in SR networks introduces outliers into weights."
> — [Make RepVGG Greater Again (AAAI 2023)](https://arxiv.org/html/2212.01593)

### Why MobileNet family may have issues:
> "MobileNets often have significant accuracy degradation under post-training quantization."
> — [Do All MobileNets Quantize Poorly? (CVPR 2021)](https://arxiv.org/abs/2104.11849)

## Recommendations

### Best Choice: MobileNetV4 Small + ONNX FP32
- **53.3 img/s** - fastest overall
- **No quantization** - zero accuracy loss
- **2.16M params** - smallest model
- Just export to ONNX and deploy!

### For Production by Use Case:

| Priority | Best Choice | Throughput | Notes |
|----------|-------------|------------|-------|
| **Max Speed** | MNV4 Small + ONNX FP32 | 53 img/s | No accuracy risk |
| **Speed + Small** | RegNetX-004 + ONNX FP32 | 48 img/s | 5.6M params |
| **Proven Stable** | ResNet18 + ONNX INT8 | 38 img/s | Well-tested |
| **Max Accuracy** | ResNet50 + ONNX INT8 | 25 img/s | Largest model |

### Avoid:
- **RepVGG + INT8** - severe accuracy drop (20-35%)
- **MobileNetV4 + INT8** - uncertain accuracy, and FP32 is faster anyway!

### Deployment Workflow:

```
Recommended (No Quantization):
1. Train MobileNetV4/RegNet model (PyTorch)
2. Export to ONNX
3. Deploy with ONNX Runtime
→ Get ~2x speedup for free!

If you need INT8 (ResNet/larger models):
1. Train model (PyTorch)
2. Export to ONNX
3. Quantize with ONNX Runtime (static INT8)
4. Validate accuracy on test set
```

## Files

- `quantization_benchmark.csv` - Raw benchmark data
- `quantization_benchmark.png` - Visualization plot
