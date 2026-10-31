"""Hardware smoke test: full warmup and first JPEG under the default IPC timeout.

Run with the controller's Python 3.10 environment on Jetson. Optional fixtures:
SPAGHETTI_TEST_ENGINE=/path/model.engine SPAGHETTI_TEST_FRAME=/path/camera.jpg
This test never connects to a printer or sends print commands.
"""
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from api.ipc_client import InferenceClient, SOCKET_PATH, recv_exact


def main():
    engine = Path(os.environ.get("SPAGHETTI_TEST_ENGINE", str(ROOT_DIR / "models/obico_spaghetti.engine")))
    if not engine.exists():
        print("[SKIP] Engine not found: {}".format(engine))
        return 0
    if os.path.exists(SOCKET_PATH):
        raise RuntimeError("Refusing to replace an existing worker socket: {}".format(SOCKET_PATH))

    frame_path = os.environ.get("SPAGHETTI_TEST_FRAME")
    if frame_path:
        jpeg_bytes = Path(frame_path).read_bytes()
    else:
        # Color/spatial differences avoid the old uniform-gray fixture's blind spot.
        image = Image.new("RGB", (1536, 1080), (180, 40, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle([400, 100, 900, 600], fill=(10, 160, 60))
        draw.rectangle([900, 0, 1279, 719], fill=(20, 30, 190))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

    model_type = os.environ.get("SPAGHETTI_TEST_TYPE")
    if not model_type:
        name = engine.name.lower()
        if "yolov5" in name:
            model_type = "yolov5"
        elif "yolov8" in name:
            model_type = "yolov8"
        else:
            model_type = "obico"

    python = ROOT_DIR / "inference/.venv-jetpack/bin/python"
    if not python.exists():
        python = Path("/usr/local/bin/jetpack-python3.6")
    cmd = [str(python), "-u", str(ROOT_DIR / "inference/worker_server.py"),
           "--engine", str(engine), "--type", model_type, "--conf", "0.25"]
    env = os.environ.copy()
    env["OPENBLAS_CORETYPE"] = "ARMV8"
    client = None
    with tempfile.TemporaryFile(mode="w+t") as log:
        started = time.monotonic()
        worker = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            # Engine load and full image/NMS warmup have their own startup budget.
            while time.monotonic() - started < 120.:
                if worker.poll() is not None:
                    raise RuntimeError("Worker exited during startup: {}".format(worker.returncode))
                if os.path.exists(SOCKET_PATH):
                    try:
                        client = InferenceClient()  # Keep the production default: 5s.
                        client.connect()
                        break
                    except (OSError, FileNotFoundError):
                        client.close()
                time.sleep(0.25)
            else:
                raise TimeoutError("Worker did not become ready within 120s")

            print(json.dumps({"stage": "ready", "startup_s": time.monotonic() - started,
                              "client_timeout_s": client.timeout}), flush=True)
            for index in range(5):
                start = time.perf_counter()
                # Inspect status explicitly: an inference error must not look
                # like a valid result with zero detections.
                client.sock.sendall(struct.pack(">I", len(jpeg_bytes)) + jpeg_bytes)
                n = struct.unpack(">I", recv_exact(client.sock, 4))[0]
                response = json.loads(recv_exact(client.sock, n).decode("utf-8"))
                elapsed_ms = (time.perf_counter() - start) * 1000.
                assert response.get("status") == "ok", response
                assert elapsed_ms < client.timeout * 1000., "Frame exceeded default IPC timeout"
                print(json.dumps({"stage": "frame", "index": index + 1,
                                  "roundtrip_ms": elapsed_ms, "response": response}), flush=True)
            print("[PASS] First frame and subsequent frames met the default IPC timeout.", flush=True)
            return 0
        finally:
            if client is not None:
                client.close()
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
            log.seek(0)
            print(log.read(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
