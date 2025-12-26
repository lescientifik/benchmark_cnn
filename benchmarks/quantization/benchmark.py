#!/usr/bin/env python3
"""Test ONNX Runtime INT8 static quantization on RepVGG + FPN.

ONNX Runtime's static INT8 keeps activations in INT8 between layers,
using optimized VNNI/AVX-512 kernels on Intel CPUs.
"""

import gc
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

from models import SegmentationModel


class RandomCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader with random data."""

    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self.current = 0

    def get_next(self):
        if self.current >= self.n_samples:
            return None
        self.current += 1
        return {"input": np.random.randn(1, 3, 512, 512).astype(np.float32)}

    def rewind(self):
        self.current = 0


def create_repvgg_fpn(num_classes=117):
    """Create and reparameterize RepVGG-A0 + FPN model."""
    model = SegmentationModel("repvgg_a0", "fpn", num_classes)
    for m in model.modules():
        if hasattr(m, "reparameterize"):
            m.reparameterize()
    return model


def export_to_onnx(model, path, input_shape=(1, 3, 512, 512)):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"   Exported to {path}")


def benchmark_onnx(session, input_shape=(1, 3, 512, 512), warmup=10, iterations=50):
    """Benchmark ONNX Runtime session."""
    gc.collect()

    x = np.random.randn(*input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: x})

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: x})
        times.append(time.perf_counter() - start)

    times_sorted = sorted(times)
    trim = len(times) // 10
    times_trimmed = times_sorted[trim:-trim] if trim > 0 else times_sorted

    return statistics.median(times_trimmed)


def benchmark_pytorch(model, warmup=10, iterations=50, use_compile=False, use_autocast=False):
    """Benchmark PyTorch model with optional compile and autocast."""
    import torch._dynamo
    torch._dynamo.reset()
    gc.collect()
    model.eval()

    if use_compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    x = torch.randn(1, 3, 512, 512)

    with torch.inference_mode():
        if use_autocast:
            with torch.autocast(device_type="cpu"):
                for _ in range(warmup):
                    _ = model(x)
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    _ = model(x)
                    times.append(time.perf_counter() - start)
        else:
            for _ in range(warmup):
                _ = model(x)
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = model(x)
                times.append(time.perf_counter() - start)

    times_sorted = sorted(times)
    trim = len(times) // 10
    times_trimmed = times_sorted[trim:-trim] if trim > 0 else times_sorted

    return statistics.median(times_trimmed)


def create_model(encoder_name, decoder="fpn", num_classes=117):
    """Create segmentation model, reparameterize if RepVGG."""
    model = SegmentationModel(encoder_name, decoder, num_classes)
    if "repvgg" in encoder_name:
        for m in model.modules():
            if hasattr(m, "reparameterize"):
                m.reparameterize()
    return model


def benchmark_model(encoder_name, decoder="fpn"):
    """Benchmark a single encoder with all modes."""
    print(f"\n{'='*70}")
    print(f"  {encoder_name.upper()} + {decoder.upper()}")
    print(f"{'='*70}")

    onnx_fp32_path = Path(f"{encoder_name}_{decoder}_fp32.onnx")
    onnx_int8_path = Path(f"{encoder_name}_{decoder}_int8.onnx")

    # Create model
    model = create_model(encoder_name, decoder)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Params: {params:.2f}M")

    # PyTorch baseline (no optimizations)
    t_pt_baseline = benchmark_pytorch(model, use_compile=False, use_autocast=False)

    # PyTorch autocast only
    model2 = create_model(encoder_name, decoder)
    t_pt_autocast = benchmark_pytorch(model2, use_compile=False, use_autocast=True)
    del model2
    gc.collect()

    # PyTorch compile + autocast
    model3 = create_model(encoder_name, decoder)
    t_pt_compiled = None
    try:
        t_pt_compiled = benchmark_pytorch(model3, use_compile=True, use_autocast=True)
    except Exception as e:
        print(f"   torch.compile failed: {e}")
    del model3
    gc.collect()

    # Export to ONNX
    try:
        export_to_onnx(model, str(onnx_fp32_path))
    except Exception as e:
        print(f"   ONNX export failed: {e}")
        return None
    del model
    gc.collect()

    # ONNX Runtime setup
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 0

    # ONNX FP32
    sess_fp32 = ort.InferenceSession(str(onnx_fp32_path), sess_options)
    t_ort_fp32 = benchmark_onnx(sess_fp32)
    del sess_fp32
    gc.collect()

    # ONNX INT8 static
    t_ort_int8 = None
    try:
        calibration_reader = RandomCalibrationDataReader(n_samples=50)
        quantize_static(
            str(onnx_fp32_path),
            str(onnx_int8_path),
            calibration_reader,
            quant_format=ort.quantization.QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
        sess_int8 = ort.InferenceSession(str(onnx_int8_path), sess_options)
        t_ort_int8 = benchmark_onnx(sess_int8)
        del sess_int8
    except Exception as e:
        print(f"   INT8 quantization failed: {e}")

    # Cleanup ONNX files
    if onnx_fp32_path.exists():
        onnx_fp32_path.unlink()
    if onnx_int8_path.exists():
        onnx_int8_path.unlink()

    gc.collect()

    return {
        "encoder": encoder_name,
        "params": params,
        "pt_baseline": t_pt_baseline,
        "pt_autocast": t_pt_autocast,
        "pt_compiled": t_pt_compiled,
        "ort_fp32": t_ort_fp32,
        "ort_int8": t_ort_int8,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ONNX Runtime INT8 Static Quantization Comparison")
    print("=" * 70)

    # Encoders to test
    encoders = [
        # ResNet family
        "resnet18",
        "resnet34",
        "resnet50",
        # RegNet family
        "regnetx_004",
        "regnetx_008",
        "regnety_004",
        "regnety_008",
        # RepVGG family
        "repvgg_a0",
        "repvgg_a1",
        # MobileNetV4 family
        "mobilenetv4_conv_small.e2400_r224_in1k",
        "mobilenetv4_conv_medium.e500_r224_in1k",
    ]

    all_results = []
    for enc in encoders:
        result = benchmark_model(enc)
        if result:
            all_results.append(result)

    # Summary table
    print("\n" + "=" * 120)
    print("SUMMARY: All Encoders + FPN (Latency in ms)")
    print("=" * 120)
    print(f"{'Encoder':<35} {'Params':<8} {'Baseline':<10} {'Autocast':<10} {'Compiled':<10} {'ORT FP32':<10} {'ORT INT8':<10} {'Best Speedup':<12}")
    print("-" * 120)

    for r in all_results:
        enc = r["encoder"]
        if len(enc) > 32:
            enc = enc[:32] + "..."

        compiled_str = f"{r['pt_compiled']*1000:.1f}" if r["pt_compiled"] else "FAIL"
        ort_int8_str = f"{r['ort_int8']*1000:.1f}" if r["ort_int8"] else "FAIL"

        # Find best time
        times = [r["pt_baseline"], r["pt_autocast"], r["ort_fp32"]]
        if r["pt_compiled"]:
            times.append(r["pt_compiled"])
        if r["ort_int8"]:
            times.append(r["ort_int8"])
        best_time = min(times)
        best_speedup = r["pt_baseline"] / best_time

        print(f"{enc:<35} {r['params']:<8.2f} {r['pt_baseline']*1000:<10.1f} {r['pt_autocast']*1000:<10.1f} {compiled_str:<10} {r['ort_fp32']*1000:<10.1f} {ort_int8_str:<10} {best_speedup:<12.2f}x")

    # Throughput comparison
    print("\n" + "=" * 120)
    print("THROUGHPUT COMPARISON (img/s)")
    print("=" * 120)
    print(f"{'Encoder':<35} {'Baseline':<10} {'Autocast':<10} {'Compiled':<10} {'ORT FP32':<10} {'ORT INT8':<10}")
    print("-" * 120)

    for r in all_results:
        enc = r["encoder"]
        if len(enc) > 32:
            enc = enc[:32] + "..."
        compiled_tp = f"{1/r['pt_compiled']:.1f}" if r["pt_compiled"] else "-"
        ort_int8_tp = f"{1/r['ort_int8']:.1f}" if r["ort_int8"] else "-"
        print(f"{enc:<35} {1/r['pt_baseline']:<10.1f} {1/r['pt_autocast']:<10.1f} {compiled_tp:<10} {1/r['ort_fp32']:<10.1f} {ort_int8_tp:<10}")

    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    df["baseline_throughput"] = 1 / df["pt_baseline"]
    df["autocast_throughput"] = 1 / df["pt_autocast"]
    df["compiled_throughput"] = df["pt_compiled"].apply(lambda x: 1/x if x else None)
    df["ort_fp32_throughput"] = 1 / df["ort_fp32"]
    df["ort_int8_throughput"] = df["ort_int8"].apply(lambda x: 1/x if x else None)
    df.to_csv("quantization_benchmark.csv", index=False)
    print("\nResults saved to quantization_benchmark.csv")

    # Create plot
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Prepare data
    encoders_short = []
    for e in df["encoder"]:
        if "mobilenetv4" in e:
            e = e.replace("mobilenetv4_conv_", "mnv4_").split(".")[0]
        encoders_short.append(e)

    x = np.arange(len(encoders_short))
    width = 0.15

    # Plot 1: Latency comparison
    ax1 = axes[0]
    ax1.bar(x - 2*width, df["pt_baseline"]*1000, width, label="PT Baseline", color="#1f77b4")
    ax1.bar(x - 1*width, df["pt_autocast"]*1000, width, label="PT Autocast", color="#ff7f0e")
    ax1.bar(x, df["pt_compiled"].fillna(0)*1000, width, label="PT Compiled", color="#9467bd")
    ax1.bar(x + 1*width, df["ort_fp32"]*1000, width, label="ONNX FP32", color="#2ca02c")
    ax1.bar(x + 2*width, df["ort_int8"].fillna(0)*1000, width, label="ONNX INT8", color="#d62728")
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Latency by Architecture and Mode\n(lower is better)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(encoders_short, rotation=45, ha="right", fontsize=9)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Throughput comparison
    ax2 = axes[1]
    ax2.bar(x - 2*width, df["baseline_throughput"], width, label="PT Baseline", color="#1f77b4")
    ax2.bar(x - 1*width, df["autocast_throughput"], width, label="PT Autocast", color="#ff7f0e")
    ax2.bar(x, df["compiled_throughput"].fillna(0), width, label="PT Compiled", color="#9467bd")
    ax2.bar(x + 1*width, df["ort_fp32_throughput"], width, label="ONNX FP32", color="#2ca02c")
    ax2.bar(x + 2*width, df["ort_int8_throughput"].fillna(0), width, label="ONNX INT8", color="#d62728")
    ax2.set_ylabel("Throughput (img/s)", fontsize=12)
    ax2.set_title("Throughput by Architecture and Mode\n(higher is better)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(encoders_short, rotation=45, ha="right", fontsize=9)
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Quantization Benchmark: Encoder + FPN @ 512x512 (CPU, batch=1)", fontsize=14)
    plt.tight_layout()
    plt.savefig("quantization_benchmark.png", dpi=150, bbox_inches="tight")
    print("Plot saved to quantization_benchmark.png")
