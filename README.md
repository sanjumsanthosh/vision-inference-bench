<div align="center">

# 📉 Vision Inference Benchmark

**MobileNetV3‑Small: PyTorch → ONNX FP32 → static INT8. Measure the speedup, then ask whether it still works.**

<br>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.31.0-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![CPU](https://img.shields.io/badge/provider-CPUExecutionProvider-lightgrey)](#measurement-conditions)
[![Accuracy](https://img.shields.io/badge/accuracy-not%20validated-critical)](#the-finding)

</div>

---

## The finding, up front

| | ONNX FP32 | ONNX INT8 | Change |
|---|---:|---:|---:|
| Median inference | 3.095 ms | **1.495 ms** | **2.07× faster** |
| p95 inference | 3.300 ms | 1.603 ms | −51% |
| Model file | 9.99 MiB | **2.69 MiB** | **−73.1%** |
| Peak process RSS | 85.375 MiB | 69.578 MiB | −18.5% |
| Sample top‑1 | 354: Arabian camel | **723: pinwheel** | ❌ **disagrees** |
| Held‑out accuracy | not measured | not measured | — |

**The quantization is fast and small, and it is not yet correct.** On the single sample image
the published INT8 graph changes the predicted class from *Arabian camel* to *pinwheel*, with a
maximum absolute logit difference of 21.43. Held‑out accuracy was never measured.

Treat the numbers above as measurements of this experiment, not a deployment claim.

---

## What this repo is

A CPU benchmark of one small vision model, end to end:

1. Load `MobileNet_V3_Small_Weights.IMAGENET1K_V1`.
2. Export ONNX FP32 with named inputs/outputs and prepare the graph.
3. Calibrate **static QDQ INT8** — signed activations/weights, per‑channel weights, MinMax ranges,
   `Conv`/`Gemm`/`MatMul`.
4. Measure FP32 vs INT8 latency with identical thread settings.
5. Compare logits, top‑1 predictions, file sizes, peak RSS, and environment details.
6. Save the run as JSON.

Two things it deliberately does **not** do: compare PyTorch against ONNX in the latency table
(that table is ONNX FP32 vs ONNX INT8), and claim accuracy it hasn't measured.

## Run the notebook

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install torch torchvision onnx onnxruntime onnxscript pillow numpy jupyterlab
python -m jupyterlab train.ipynb
```

In the first cell, set:

| Input | Meaning | Default |
|---|---|---|
| `sample_image_path` | one local image | `~/Downloads/camel.jpg` (not bundled) |
| `calibration_dir` | representative `.jpg` files | `/tmp/onnx/calibration` (not bundled) |
| `warmup_runs` / `measured_runs` | timing loop | 20 / 200 |
| thread counts | ONNX Runtime intra/inter‑op | 1 / 1 |
| `eval_samples` | optional `(image_path, imagenet_class_index)` pairs | none |

Run top to bottom. Outputs are cleared in the saved notebook so old measurements can't be mistaken
for new ones, and missing input data raises a clear error instead of silently skipping.

The recorded run calibrated on **200 images from [Imagenette](https://github.com/fastai/imagenette)** —
20 each from ten classes, 320 px training split. That is a starter subset, not ImageNet coverage.

### Outputs

```text
artifacts/
  mobilenet_fp32.onnx            # exported FP32 graph, weights embedded
  pre_quant_mobilenet_fp32.onnx  # prepared FP32 graph
  mobilenet_int8.onnx            # static QDQ INT8 graph
  benchmark_results.json         # this run's measurements + conditions
```

An unmeasured metric is `null`, never `0`. The root‑level `benchmark_results.json` is the
published snapshot from 23 Oct 2026; new runs write under `artifacts/`.

## Measurement conditions

| Setting | Recorded value |
|---|---|
| CPU / OS | Apple M4 Pro / macOS 26.7.1, arm64 |
| Provider | `CPUExecutionProvider` |
| Input | batch 1, float32 `[1, 3, 224, 224]` |
| Threads | intra 1 / inter 1, sequential execution |
| Warm‑up / measured | 20 / 200 |
| Python · PyTorch · ORT | 3.14.0 · 2.14.1 · 1.31.0 |

Median is the middle latency, p95 the 95th percentile. Timing covers `session.run` on one
preprocessed image and excludes loading, preprocessing, export, and session creation. Peak RSS is
the whole process in a fresh run — interpreter and runtime included, not model memory alone.
Every one of those choices moves the number; re‑measure before quoting.

## Accuracy: what's actually known

- The camel → pinwheel disagreement needs bisecting across PyTorch, exported FP32, prepared FP32,
  and INT8 before anyone claims "accuracy‑preserving".
- Use separate labelled images for evaluation. Imagenette's `0–9` labels must be mapped to the
  model's ImageNet indices `0–999`; the current guard rejects reused calibration paths but cannot
  catch an image copied to another filename.
- One matching top‑1 prediction would still not establish accuracy.
- Calibration and sample images aren't bundled, and the snapshot has no artifact/data hashes, so
  exact reproduction isn't possible from this repo alone.

## Scope

CPU inference only. This does **not** establish GPU/DSP, Snapdragon or headset performance, power
draw, or end‑to‑end camera latency. C++ ONNX Runtime parity and a Unity/OpenXR pendulum project are
planned, not implemented.

## Author

[@sanjumsanthosh](https://github.com/sanjumsanthosh)
