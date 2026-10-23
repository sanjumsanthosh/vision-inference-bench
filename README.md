# Vision Inference Benchmark

A small CPU benchmark for MobileNetV3-Small: load pretrained PyTorch weights,
export ONNX FP32, prepare the graph, calibrate static INT8 quantization, and compare
latency, saved model size, process memory, and numerical outputs.

**Status:** the Python benchmark is implemented. The published INT8 run changes
the sample prediction from **Arabian camel to pinwheel**. Held-out accuracy is not
measured, so the INT8 model has **not passed correctness validation**. Performance
improvements below are measurements of this experiment, not a deployment claim.

## Run the notebook

Use a Python environment compatible with PyTorch and ONNX Runtime. The recorded
environment is listed below; dependency versions are not pinned by this repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision onnx onnxruntime onnxscript pillow numpy jupyterlab
python -m jupyterlab train.ipynb
```

In the first code cell, configure:

- `sample_image_path`: one local image. The current convenience default is
  `~/Downloads/camel.jpg`; this file is not included.
- `calibration_dir`: a directory of representative `.jpg` calibration images.
  The current local default is `/tmp/onnx/calibration`; no images are downloaded
  automatically or bundled in the repository.
- `warmup_runs`, `measured_runs`, and thread counts: shared by measurement and reporting.
- `eval_samples`: optional `(image_path, imagenet_class_index)` pairs for held-out accuracy.

Run the notebook from the repository directory, top to bottom, using its Python
environment. Outputs are cleared in the saved notebook so old measurements are
not displayed as if they came from edited code. Missing input data raises a clear error.

For the recorded experiment, calibration used 200 images from the
[Imagenette dataset](https://github.com/fastai/imagenette), 20 from each of ten classes,
downloaded from the `ShaomuTan/imagenette` Hugging Face dataset's 320px training split.
This is a starter subset, not full ImageNet coverage or an XR deployment distribution.
For reproduction, supply the same images if retained, or document a new calibration
set and treat the run as a different experiment. `/tmp` data is temporary.

## Notebook workflow

1. Configure inputs, output directory, and benchmark settings.
2. Load `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, set evaluation mode, and apply its image transforms.
3. Export ONNX with named inputs/outputs and prepare the FP32 graph.
4. Calibrate static QDQ quantization using signed INT8 activations/weights,
   per-channel weights, MinMax ranges, and selected `Conv`, `Gemm`, and `MatMul` operations.
5. Measure FP32 and INT8 ONNX Runtime CPU inference with matching thread settings.
6. Compare logits, top-1 predictions, file sizes, and environment details.
7. Optionally evaluate labeled held-out images.
8. Measure each model's peak whole-process RSS in a separate process (macOS/Linux).
9. Save the run as JSON.

The notebook prints PyTorch predictions as a baseline, but the latency table
compares **ONNX FP32 against ONNX INT8**, not PyTorch against ONNX.

## Recorded sample results

These values come from the checked-in [benchmark_results.json](benchmark_results.json),
recorded on 23 October 2026. A new run writes a separate generated result under
`artifacts/`; the table is a historical snapshot and is not updated automatically.

| Metric | ONNX FP32 | ONNX INT8 |
|---|---:|---:|
| Median inference | 3.095 ms | 1.495 ms |
| p95 inference | 3.300 ms | 1.603 ms |
| Model file | 10,475,490 bytes / 9.99 MiB | 2,820,172 bytes / 2.69 MiB |
| Peak process RSS | 85.375 MiB | 69.578 MiB |
| Sample top-1 class | 354: Arabian camel | 723: pinwheel |
| Held-out accuracy | Not measured | Not measured |

- Median speedup: **2.07x** (`FP32 median / INT8 median`).
- File size reduction: **73.1%**.
- Peak process RSS reduction: **18.5%**.
- Maximum absolute logit difference: **21.4303**; mean absolute difference: **3.3181**.
  These are raw-score differences, not accuracy-loss percentages.

### Measurement conditions

| Setting | Recorded value |
|---|---|
| CPU / OS | Apple M4 Pro / macOS 26.7.1, arm64 |
| Runtime provider | CPUExecutionProvider |
| Batch / input | 1 / float32 `[1, 3, 224, 224]` |
| Intra-op / inter-op threads | 1 / 1; sequential graph execution |
| Warm-up / measured runs | 20 / 200 |
| Python | 3.14.0 |
| PyTorch / TorchVision | 2.14.1 / 0.29.1 |
| ONNX Runtime | 1.31.0 |

Median is the middle latency; p95 is the 95th percentile of measured latencies.
Timing repeatedly uses one preprocessed image and includes `session.run` overhead,
but excludes file loading, preprocessing, export, and session creation. Peak RSS
includes interpreter/runtime overhead and loading; it is not model memory alone.
Measurements vary with hardware, runtime versions, scheduling, and calibration data.

## Generated outputs

With the notebook launched from this directory:

```text
artifacts/
  mobilenet_fp32.onnx            # exported FP32 graph with embedded weights
  pre_quant_mobilenet_fp32.onnx  # prepared FP32 graph
  mobilenet_int8.onnx            # static QDQ INT8 graph
  benchmark_results.json        # current run's measurements and conditions
```

The JSON contains `fp32`/`int8` latency and file bytes, `median_speedup`,
`file_size_reduction_pct`, `single_image_comparison`, `conditions`, `accuracy`, and
`memory`. An unmeasured metric is `null`, not zero. The memory cell updates the
in-memory results dictionary; rerun the final save cell after remeasuring.

Generated artifacts and notebook checkpoints are ignored by Git. The root-level
`benchmark_results.json` remains the published sample run, including its original
local artifact paths. Those historical paths do not configure new runs.

## Accuracy and current limitations

- The camel-to-pinwheel disagreement needs investigation. Compare PyTorch,
  exported FP32, prepared FP32, and INT8 outputs to locate the divergence before
  claiming accuracy-preserving quantization.
- Use separate labeled evaluation images. Imagenette labels `0-9` must be mapped
  to the model's ImageNet class indices `0-999`. The current guard rejects reused
  calibration paths, but cannot detect an image copied to another filename.
- A matching top-1 prediction on one image would still not establish accuracy.
- The calibration images and sample image are not bundled, and the historical
  snapshot does not contain artifact/data hashes for exact reproduction.
- This project currently measures CPU inference. It does not establish GPU/DSP,
  Snapdragon/headset performance, power use, or end-to-end camera latency.

C++ ONNX Runtime parity and a separate Unity/OpenXR pendulum project remain planned;
they are not implemented features of this repository.
