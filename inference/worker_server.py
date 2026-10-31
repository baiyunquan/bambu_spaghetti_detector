#!/usr/bin/env python3
"""
GPU Inference Worker Server (Unix Domain Socket)
=================================================
Runs in JetPack Python 3.6 environment with full CUDA/TensorRT acceleration.
Listens on /tmp/spaghetti_trt.sock for raw JPEG frames, executes TensorRT
inference with GPU NMS, and returns JSON detection results.
"""

import os
import sys
import time
import json
import socket
import struct
import argparse
import errno
import stat
import signal
from pathlib import Path

# Mandatory Jetson Nano OpenBLAS setting
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

# Add current folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

SOCKET_PATH = "/tmp/spaghetti_trt.sock"
MAX_JPEG_BYTES = 5 * 1024 * 1024


def recv_exact(sock, num_bytes: int) -> bytes:
    """Reads exactly num_bytes from a socket."""
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def detect_frame(detector, preprocessor, postprocessor, model_type, jpeg_bytes):
    """Use the same complete path for startup warmup and client requests."""
    tensor, meta = preprocessor.decode_and_preprocess(jpeg_bytes)
    outputs = detector.infer(tensor)
    if model_type == "obico":
        return postprocessor.process_obico(outputs, meta, detector.class_names)
    elif model_type == "yolov5":
        return postprocessor.process_yolov5(outputs, meta, detector.class_names)
    return postprocessor.process_yolov8(outputs, meta, detector.class_names)


def warmup_pipeline(detector, preprocessor, postprocessor, model_type):
    """Finish lazy image/CUDA/NMS initialization before publishing readiness."""
    import cv2
    import numpy as np
    import torch

    # Cover nominal 720p and the actual A1's 1536x1080 stream. The latter
    # triggers a different OpenCV resize/thread-pool path on this Jetson;
    # prewarming only 720p still leaves a multi-second first real request.
    for source_h, source_w in ((720, 1280), (1080, 1536)):
        image = np.random.RandomState(0).randint(0, 256, (source_h, source_w, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Failed to encode warmup JPEG")
        jpeg_bytes = encoded.tobytes()
        del image, encoded
        for _ in range(3):
            detect_frame(detector, preprocessor, postprocessor, model_type, jpeg_bytes)

    # The model may return no candidates for the warmup image. Force the full
    # nonempty GPU NMS + coordinate conversion path without affecting decisions.
    tensor, meta = preprocessor.decode_and_preprocess(jpeg_bytes)
    device = tensor.device
    if model_type == "obico":
        boxes = torch.tensor([[[[0.1, 0.1, 0.5, 0.5]],
                               [[0.12, 0.12, 0.48, 0.48]]]], device=device)
        scores = torch.ones((1, 2, 1), device=device)
        postprocessor.process_obico([boxes, scores], meta, detector.class_names)
    elif model_type == "yolov5":
        dummy = torch.zeros((1, 2, 5 + len(detector.class_names)), device=device)
        dummy[0, :, :4] = torch.tensor([[100., 100., 50., 50.], [102., 102., 50., 50.]], device=device)
        dummy[0, :, 4] = 1.0
        spag_idx = detector.class_names.index("spaghetti") if "spaghetti" in detector.class_names else 0
        dummy[0, :, 5 + spag_idx] = 1.0
        postprocessor.process_yolov5([dummy], meta, detector.class_names)
    else:
        outputs = torch.zeros((1, 4 + len(detector.class_names), 2), device=device)
        outputs[0, :4, :] = torch.tensor([[100., 102.], [100., 102.],
                                         [50., 50.], [50., 50.]], device=device)
        outputs[0, 4, :] = 1.0
        postprocessor.process_yolov8([outputs], meta, detector.class_names)
    if tensor.is_cuda:
        torch.cuda.synchronize(device)


class LegacyPipeline:
    """Keep the existing torch path behind an explicit lazy boundary."""
    def __init__(self, engine_path, model_type, conf_thresh, iou_thresh):
        from gpu_preprocess import GPUPreprocessor
        from postprocess import GPUPostProcessor
        from trt_detector import TensorRTDetector

        self.model_type = model_type
        self.detector = TensorRTDetector(engine_path, model_type=model_type)
        input_h, input_w = self.detector.get_input_resolution()
        self.preprocessor = GPUPreprocessor(target_size=(input_h, input_w))
        self.postprocessor = GPUPostProcessor(conf_thresh=conf_thresh, iou_thresh=iou_thresh)

    def detect(self, jpeg_bytes):
        return detect_frame(self.detector, self.preprocessor, self.postprocessor,
                            self.model_type, jpeg_bytes)

    def warmup(self):
        warmup_pipeline(self.detector, self.preprocessor, self.postprocessor, self.model_type)

    def close(self):
        # The legacy objects release their owned tensors/context on destruction.
        self.postprocessor = self.preprocessor = self.detector = None


def create_pipeline(engine_path, model_type, conf_thresh, iou_thresh, backend="torch"):
    if backend == "torch":
        return LegacyPipeline(engine_path, model_type, conf_thresh, iou_thresh)
    if backend == "native_cuda":
        from native_cuda.pipeline import NativePipeline
        return NativePipeline(engine_path, model_type, conf_thresh, iou_thresh)
    raise ValueError("Unsupported inference backend: {}".format(backend))


def remove_stale_socket(path):
    """Preserve live listeners and unrelated files; remove refused UDS only."""
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(before.st_mode):
        raise RuntimeError("Refusing to replace non-socket path: {}".format(path))
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.25)
        try:
            probe.connect(path)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return
            if exc.errno != errno.ECONNREFUSED:
                raise RuntimeError("Cannot safely probe existing socket: {}".format(exc))
        else:
            raise RuntimeError("Inference socket is already active: {}".format(path))
    finally:
        probe.close()
    try:
        after = os.lstat(path)
    except FileNotFoundError:
        return
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise RuntimeError("Inference socket changed during stale-socket probe")
    os.unlink(path)


def _send_response(conn, response):
    payload = json.dumps(response, allow_nan=False).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def serve_connection(conn, pipeline):
    """Handle bounded JPEG requests; inference errors never become good frames."""
    while True:
        header = recv_exact(conn, 4)
        if not header:
            return
        payload_len = struct.unpack(">I", header)[0]
        if not 0 < payload_len <= MAX_JPEG_BYTES:
            _send_response(conn, dict(status="error", error="Invalid JPEG payload length", detections=[]))
            return
        jpeg_bytes = recv_exact(conn, payload_len)
        if not jpeg_bytes:
            return
        started = time.monotonic()
        try:
            detections = pipeline.detect(jpeg_bytes)
            response = dict(status="ok", detections=detections,
                            inference_time_ms=round((time.monotonic() - started) * 1000.0, 2))
            # Include serialization validation in the error boundary so a NaN
            # result cannot masquerade as an empty, healthy detection frame.
            json.dumps(response, allow_nan=False)
        except Exception as exc:
            response = dict(status="error", error=str(exc), detections=[])
        _send_response(conn, response)


def run_worker(engine_path: str, model_type: str = "obico", conf_thresh: float = 0.35,
               iou_thresh: float = 0.45, backend: str = "torch"):
    print("[INFO] Starting {} worker: {} ({})".format(backend, engine_path, model_type), flush=True)
    pipeline = create_pipeline(engine_path, model_type, conf_thresh, iou_thresh, backend)
    server_sock = None
    bound_identity = None
    try:
        print("[INFO] Warming up JPEG preprocessing, TensorRT and GPU NMS...", flush=True)
        pipeline.warmup()
        remove_stale_socket(SOCKET_PATH)
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(SOCKET_PATH)
        info = os.lstat(SOCKET_PATH)
        bound_identity = (info.st_dev, info.st_ino)
        server_sock.listen(5)
        os.chmod(SOCKET_PATH, 0o777)
        print("[INFO] Full pipeline warmup complete. Listening for connections...", flush=True)
        while True:
            conn, _ = server_sock.accept()
            try:
                serve_connection(conn, pipeline)
            except (OSError, ConnectionError) as exc:
                print("[WARN] Client disconnected: {}".format(exc), flush=True)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[INFO] Worker shutting down...")
    finally:
        try:
            if server_sock is not None:
                server_sock.close()
            if bound_identity is not None:
                try:
                    info = os.lstat(SOCKET_PATH)
                    if bound_identity == (info.st_dev, info.st_ino):
                        os.unlink(SOCKET_PATH)
                except FileNotFoundError:
                    pass
        finally:
            pipeline.close()
        print("[INFO] Worker stopped.", flush=True)


def _terminate(signum, frame):
    raise KeyboardInterrupt()


def main():
    parser = argparse.ArgumentParser(description="TensorRT Spaghetti Detection UDS Worker")
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine plan file")
    parser.add_argument("--type", type=str, default="obico", choices=["obico", "yolov8", "yolov5"], help="Model architecture")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--backend", default="torch", choices=["torch", "native_cuda"], help="Inference backend (no automatic fallback)")
    parser.add_argument("--test-dry", action="store_true", help="Perform engine load test and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _terminate)
    if args.test_dry:
        pipeline = create_pipeline(args.engine, args.type, args.conf, args.iou, args.backend)
        try:
            pipeline.warmup()
            print("[TEST] Engine and full pipeline warmup succeeded.")
        finally:
            pipeline.close()
        return

    run_worker(
        engine_path=args.engine,
        model_type=args.type,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        backend=args.backend
    )


if __name__ == "__main__":
    main()
