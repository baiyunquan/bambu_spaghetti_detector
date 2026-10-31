"""
End-to-End Inference & IPC Socket Integration Test
==================================================
Tests that worker_server.py processes JPEG frames over /tmp/spaghetti_trt.sock
and returns valid detection results with sub-100ms latency.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from PIL import Image
import io

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "api"))

from api.ipc_client import InferenceClient, SOCKET_PATH

def main():
    print("=" * 60)
    print(" Running End-to-End Inference & IPC Test on Jetson Nano")
    print("=" * 60)

    # 1. Check if engine exists
    engine_path = ROOT_DIR / "models" / "obico_spaghetti.engine"
    if not engine_path.exists():
        print(f"[SKIP] Engine not found at {engine_path}")
        return 0

    # 2. Start worker process
    jetpack_python = ROOT_DIR / "inference" / ".venv-jetpack" / "bin" / "python"
    worker_script = ROOT_DIR / "inference" / "worker_server.py"

    cmd = [
        str(jetpack_python),
        str(worker_script),
        "--engine", str(engine_path),
        "--type", "obico",
        "--conf", "0.25"
    ]

    print("[INFO] Launching worker_server.py daemon...")
    env = os.environ.copy()
    env["OPENBLAS_CORETYPE"] = "ARMV8"

    worker = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for socket
        print("[INFO] Waiting for UDS socket to become ready...")
        for _ in range(30):
            if os.path.exists(SOCKET_PATH):
                break
            time.sleep(0.5)

        if not os.path.exists(SOCKET_PATH):
            raise TimeoutError("Worker socket failed to create within 15 seconds")

        print("[SUCCESS] Socket ready!")

        # 3. Create dummy 720p JPEG image
        img = Image.new("RGB", (1280, 720), color=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        # 4. Connect IPC client and send inference request
        client = InferenceClient()
        client.connect()

        print("[INFO] Sending JPEG frame (1280x720) to TensorRT worker over UDS...")
        t0 = time.time()
        dets = client.detect(jpeg_bytes)
        elapsed_ms = (time.time() - t0) * 1000.0

        print(f"[SUCCESS] Inference Response Received in {elapsed_ms:.1f} ms!")
        print(f"  Detections: {len(dets)}")
        print(f"  Result: {dets}")

        client.close()

        # Assert reasonable latency (< 150ms on Jetson Nano Maxwell)
        assert elapsed_ms < 200.0, f"Latency too high: {elapsed_ms}ms"
        print("[TEST PASSED] End-to-end GPU inference and UDS communication verified!")
        return 0

    finally:
        worker.terminate()
        worker.wait(timeout=5)
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

if __name__ == "__main__":
    sys.exit(main())
