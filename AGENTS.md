# AGENTS.md - Bambu Lab A1 Spaghetti Failure Detection and Print Halting System Specification

This project runs on NVIDIA Jetson Nano (L4T R32.7.6 / JetPack 4.6). It captures the real-time video stream of a Bambu Lab A1 3D printer, detects "Spaghetti Failure" and other print failures using deep learning models, and immediately sends pause/stop commands to the printer when sufficient evidence is confirmed.

All AI Agents and development work participating in this project must strictly comply with the following specifications and constraints:

---

## 1. Dual-Runtime Architecture and Python Environment Isolation

To achieve maximum performance and seamless API compatibility, the system is strictly split into two decoupled processes communicating via **Unix Domain Socket (UDS)**:

1. **Main Controller & Bambu API Process (`/usr/bin/python3` -> Python 3.10)**:
   - Runs in the system's standard Python 3.10 environment.
   - Responsible for Bambu Cloud authentication, device discovery, TLS port 6000 video stream receiving, MQTT status monitoring, multi-frame failure state machine, and print pause/stop dispatching.
   - Natively compatible with modern Python syntax and the `bambu-lab-cloud-api` package.
2. **GPU Inference Worker Process (`/usr/local/bin/jetpack-python3.6` -> Python 3.6.9)**:
   - Runs in the specialized JetPack Python 3.6 environment equipped with **CUDA 10.2**, **TensorRT 8.2.1.9**, **PyTorch 1.10.0 (CUDA)**, and CUDA-enabled **OpenCV 4.10.0**.
   - Listens on `/tmp/spaghetti_trt.sock`. Receives raw JPEG image payloads, executes GPU-accelerated preprocessing, runs the TensorRT FP16 engine, applies GPU NMS, and returns detection results (boxes, scores, labels) as JSON.
   - **Mandatory OpenBLAS Environment Variable**:
     When executing `jetpack-python3.6` on Jetson Nano (Cortex-A57), `OPENBLAS_CORETYPE=ARMV8` must be set:
     ```bash
     export OPENBLAS_CORETYPE=ARMV8
     ```
     Failure to set this triggers an `Illegal instruction (core dumped)` exception when importing NumPy or OpenCV.
3. **Inter-Process Communication (IPC)**:
   - Uses a local Unix Domain Socket at `/tmp/spaghetti_trt.sock`.
   - Protocol: 4-byte big-endian payload length header followed by binary JPEG bytes; returns compact JSON response. Latency overhead is strictly < 1ms.

---

## 2. Model Zoo, Conversion Pipeline & TensorRT FP16 Inference

The system abstracts a unified `TensorRTDetector` interface to support hot-swapping between multiple failure detection models via `config.json`:

1. **Supported Model Architectures**:
   - **Obico (The Spaghetti Detective)**: Battle-tested dedicated spaghetti failure detection model trained on millions of real-world 3D printing failure frames.
   - **YOLOv8n Failure Model**: Compact YOLOv8 nano model trained on printing failures (spaghetti and warping).
   - **YOLOv5s Failure Model**: YOLOv5 lightweight failure detector.
2. **Two-Stage Model Conversion Pipeline**:
   - **Stage 1 (Host PC with NVIDIA RTX GPU)**:
     - Export PyTorch weights (`.pt`) to ONNX format using modern Ultralytics tools:
       ```bash
       yolo export model=best.pt format=onnx opset=11 simplify=True imgsz=640
       ```
     - Output models must use fixed input dimensions (e.g., 416x416 or 640x640) to eliminate dynamic shape profiling overhead.
   - **Stage 2 (Jetson Nano Target Compilation)**:
     - Build hardware-specific `.engine` plans directly on Jetson Nano using its native `trtexec`:
       ```bash
       /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
       ```
     - **FP16 Optimization Rule**: FP16 provides ~2x speedup on Jetson Nano Maxwell GPU. INT8 is strictly avoided as Maxwell lacks INT8 Tensor Cores and causes accuracy degradation without speedup.
3. **GPU-Accelerated Post-Processing (NMS)**:
   - For models whose ONNX does not bundle `EfficientNMS_TRT`, post-processing must use PyTorch CUDA tensors and `torchvision.ops.nms` on GPU to avoid slow Python CPU loops on the Cortex-A57 CPU.

---

## 3. Video Preprocessing and GPU Acceleration Pipeline

The Bambu Lab A1 camera produces ~0.5 to 2 FPS (1280x720 JPEG frames):

1. **Preprocessing Pipeline**:
   - JPEG bytes are decoded into host memory using OpenCV.
   - Decoded frames are immediately uploaded to GPU memory as PyTorch CUDA Tensors (or `cv2.cuda_GpuMat`).
   - Color space conversion (BGR -> RGB), Letterbox aspect-ratio preserving resize, padding, and normalization (`/ 255.0`) are executed entirely within GPU memory.
   - Preprocessed GPU buffers map directly to TensorRT input binding pointers, eliminating unnecessary CPU-GPU memory copies.
2. **Resource & Memory Budget**:
   - Jetson Nano has 4GB shared unified memory. Intermediate tensors and frame buffers must be explicitly recycled to guarantee zero memory leaks over weeks of continuous operation.

---

## 4. Device Identification, Network Topology, and Control Communication

1. **Multi-device Identification and Topology Isolation**:
   - The network reaches the `192.168.3.0/24` subnet where two Bambu Lab printers reside:
     - **Target A1 Printer**: IP is `192.168.3.231`, Serial Number prefix is `039` (SN: `03900D6A1521014`).
     - **Non-target P1S Printer**: IP is `192.168.3.164`, Serial Number prefix is `01P` (SN: `01P00C550801214`).
   - **Strict Constraint**: Device binding during initialization must strictly validate the serial number prefix (`039`) to prohibit accidentally connecting to or controlling the P1S.
2. **Cloud Account Mode and Credential Acquisition**:
   - User configures Bambu Cloud credentials (email, password) and service region (`china` or `global`) in `config.json`.
   - On startup, the Python 3.10 controller calls the Bambu Cloud API to authenticate, retrieve bound devices, and automatically extract the A1's `dev_access_code` (8-digit LAN access code) and device IP.
3. **Video Stream Acquisition Channel**:
   - Once `dev_access_code` is obtained, connect directly to the A1's local TCP port 6000 (TLS encrypted, Bambu frame protocol) to stream real-time 1280x720 JPEG images with minimal latency and high stability.
4. **Printer Control Channel (MQTT)**:
   - Issue control commands via local MQTT (port 8883, TLS) or Cloud MQTT.
   - **Pause Printing (Default Recommended)**:
     ```json
     {"print": {"sequence_id": "0", "command": "pause"}}
     ```
   - **Stop Printing (Fallback Configuration)**:
     ```json
     {"print": {"sequence_id": "0", "command": "stop"}}
     ```

---

## 5. Failure Decision Logic and Safety Strategies (Anti-False-Alarm)

1. **Multi-frame Temporal Confidence Verification**:
   - Triggering a halt based on a single isolated detection is strictly prohibited.
   - Employ a sliding window or consecutive detection mechanism: halt is only confirmed when detection confidence exceeds the threshold for N consecutive frames (default 6~8 frames).
2. **State Interlocking & False Alarm Prevention**:
   - Only evaluate failure detection when the printer is actively printing (`gcode_state == "RUNNING"`).
   - Automatically bypass detection during first layer calibration, bed leveling, filament change, or paused state.
3. **Forensic Evidence & Logging**:
   - When a pause is triggered, render bounding boxes and confidence scores on the current frame and save to `snapshots/failure_YYYYMMDD_HHMMSS.jpg`.
   - Output detailed decision logs (consecutive hit count, highest confidence, timestamp) to console and log files.

---

## 6. Development Workflow, Git Discipline & Autonomous Execution

1. **Prompt, Atomic Git Commits**:
   - Commit code changes promptly and incrementally in small, self-contained, atomic units as each feature, fix, or test is verified.
   - Strictly avoid lumping unrelated changes, multi-module refactors, or modifications originating from different agents/tools into a single ambiguous batch commit.
2. **Autonomous Execution & Decisive Proactivity**:
   - Act decisively and execute necessary operations (e.g., building engines, running tests, committing code, switching configurations, downloading datasets) proactively without constantly pausing to ask the user for routine permission or confirmation.
   - When a clear technical path or instruction is given, execute the full cycle end-to-end and report concrete results directly.
