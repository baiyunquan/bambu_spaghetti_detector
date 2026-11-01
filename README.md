# Bambu Spaghetti Failure Detection & Print Halting System

Real-time spaghetti failure detection and print halting system for Bambu Lab A1 3D printer running on NVIDIA Jetson Nano.

## Architecture
- `api/`: Main controller and Bambu API service (Python 3.10)
- `inference/`: GPU inference worker with TensorRT FP16 (JetPack Python 3.6)
- `scripts/`: Utility scripts (Access Code retrieval, model conversion)

## Inference readiness and validation

The preprocessor produces contiguous RGB NCHW FP32 tensors. The detector checks
shape and dtype and materializes strided inputs before passing pointers to
TensorRT's linear bindings.

The worker prewarms JPEG decoding, preprocessing, TensorRT, and nonempty GPU NMS
before opening its socket, covering both nominal 1280x720 and the observed A1
stream size of 1536x1080. The controller allows up to 120 seconds for startup
and verifies that the socket accepts connections. Per-frame IPC retains its
default 5-second timeout.

Run the tests in both deployed environments; cases requiring the other runtime
are skipped automatically:

```bash
api/.venv/bin/python -m unittest discover -s tests -v
OPENBLAS_CORETYPE=ARMV8 inference/.venv-jetpack/bin/python -m unittest discover -s tests -v
```

On Jetson, run the separate hardware smoke test with no other worker running:

```bash
api/.venv/bin/python tests/test_e2e_inference.py
```

Set `SPAGHETTI_TEST_FRAME=/path/camera.jpg` to replay a real JPEG, and optionally
`SPAGHETTI_TEST_ENGINE=/path/model.engine` to select an Obico engine. This test
uses the default IPC timeout, checks the first and subsequent responses, cleans
up its own worker, and never connects to a printer. The color/layout regression
checks verify tensor storage and model output consistency; they do not establish
failure-detection accuracy on real prints.

## Native CUDA backend (validation required before enabling)

`detection.backend` defaults to `"torch"`; the worker also accepts
`--backend torch` or `--backend native_cuda`. The native implementation uses
TensorRT FP16, small CUDA preprocessing kernels, and CUDA decode/class-aware NMS.
It imports no torch or torchvision and never falls back to the legacy backend
if initialization, metadata checks, or inference fail. Obico, YOLOv8, and
YOLOv5 retain their original class IDs. The known YOLOv8 export name
`Spaghetti - v11 2025-04-15 12-28am` maps to `spaghetti`; other labels remain
unchanged. The requested failure targets are `spaghetti`, `warping`, and `error`.

OpenCV still decodes JPEG on the host. Native kernels perform resize, padding,
RGB conversion, normalization and continuous NCHW packing on the GPU. Nano
profiling found that initializing OpenCV CUDA adds about 500 MiB of eager module
memory on CUDA 10.2, so the native backend retains the device primary context
and owns one CUDA Driver stream instead. It neither calls OpenCV CUDA nor
creates another context; dimensions change only the current source buffer.

Compile each fixed-shape ONNX engine on the Jetson using its native TensorRT
FP16 toolchain. Build the CUDA kernel PTX once as a separate preparation step,
with no inference worker running; NVRTC is used offline, outside worker startup:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
OPENBLAS_CORETYPE=ARMV8 /usr/local/bin/jetpack-python3.6 inference/native_cuda/build_kernels.py
/usr/bin/python3 scripts/write_model_metadata.py --engine model.engine --type yolov5
```

The build writes `inference/native_cuda/build/kernels.ptx` and its source/PTX
hash manifest. Keep both beside the deployed native code. The metadata command
writes `model.engine.meta.json` with schema version, streaming engine SHA256,
model type, and original labels. For a different export, run the metadata writer
on a host with `onnx` installed and add `--onnx source.onnx` to read its original
`names` metadata safely. Without that option, labels come from the known Obico,
YOLOv8, or YOLOv5 model defaults. Preserve the sidecar alongside the exact engine;
missing metadata, a changed engine hash, incompatible binding shape/class count,
or nonfinite inference values cause an error.

Keep `"backend": "torch"` until all validation gates pass: all five engines and
295 held-out images must agree with the corrected reference (RGB error at most
1/255, boxes at most 1 pixel, scores at most 1e-3, and the same failure and
multi-frame decisions). Steady RSS must be at most 1.5 GiB, peak RSS at most
1.8 GiB, with at least 800 MiB saved versus the runnable baseline; steady P95
latency must be within 110% of baseline. Run separate one-hour soaks for
Obico416, YOLOv8 640, and YOLOv5 640: steady RSS/PSS growth at most 16 MiB and
no increasing swap. Verify no torch/torchvision modules or libtorch mappings
on the native path. Hash-check deployed artifacts before explicitly switching
the backend. Stop the existing worker before changing backends because the
controller reuses an already active socket. Keep dry-run enabled while validating.
The startup allowance remains 120 seconds and the first frame must finish within
the existing 5-second IPC timeout.

Native warmup completes before the worker binds its socket. The worker refuses
to replace an active socket or an unrelated filesystem entry. UDS requests retain
the 4-byte big-endian length header and JPEG payload, limited to 5 MiB as in the
camera protocol. Invalid payload lengths or inference failures return
`status: "error"`; the controller raises the error and reconnects rather than
treating it as an empty successful detection frame.

The CPU contract/backend regressions use pytest, without connecting to a printer:

```bash
python3 -m pytest -q tests/test_model_contract.py tests/test_backend_selection.py
```
