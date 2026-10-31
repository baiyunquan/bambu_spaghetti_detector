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
