# Three-model native CUDA memory optimization

Approved scope: Python 3.10 controller, Python 3.6 worker, unchanged UDS and
printer control; Obico, YOLOv8 and YOLOv5 without importing torch/torchvision
on the native path. Source plan is the user's approved implementation request.

## Tasks
1. Model contracts: explicit shapes, SHA256 metadata, canonical class names,
   NumPy reference decoder/NMS, and regression tests. ModelMetadata uses
   engine_path + '.meta.json'; labels preserve original class IDs.
2. Native CUDA runtime: ctypes primary-context interoperability, reusable device
   buffers and stream, OpenCV GPU planar NCHW preprocessing, NVRTC-built sm53
   decode/sort/bitmask/greedy NMS kernels with no top-k truncation. GPU tests.
3. Integration: lazy backend selection (default torch), native warmup before
   socket bind, metadata/finite/error checks, controlled cleanup, docs/AGENTS.
4. Verification: all five engines, 295 held-out images, serial old/new benchmarks,
   one-hour soak each on Obico416/YOLOv8640/YOLOv5640, independent review,
   then hash-checked deployment and explicit backend switch only if gates pass.

## Binding interfaces
ModelContract(engine_path, model_type): class_names, canonical_labels,
validate_bindings(input_shape, output_shapes), read/write SHA256 metadata.
DeviceTensor(ptr, shape, dtype, owner): nbytes; no PyTorch types.
NativePipeline(engine_path, model_type, conf_thresh, iou_thresh): detect(bytes),
warmup(), close(). TensorRT raw output pointers never round-trip through host.

## Gates
Confidence >= threshold; per-anchor highest class; v5 objectness times class;
NMS suppresses same class only at IoU > threshold; ties by anchor index.
Only spaghetti/warping are default failure targets. Same weights and thresholds.
RGB/preprocessing error <=1/255; boxes <=1px and scores <=1e-3 versus corrected
reference, same failure and multi-frame decisions on 295 images.
Steady RSS <=1.5GiB; peak <=1.8GiB; >=800MiB saving against runnable baseline;
first frame <5s; steady P95 <=110% baseline; startup budget120s.
Three serial one-hour soaks; steady RSS/PSS growth <=16MiB; swap not increasing.
No torch/torchvision/libtorch modules or mapped libraries on native path.
No INT8, model shrinking, training, new printer controls or Rust rewrite.
