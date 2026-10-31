#!/usr/bin/env python3
"""
Bambu Lab A1 Spaghetti Failure Detection & Print Halting System
===============================================================
Main Orchestrator (Python 3.10)
Coordinates cloud authentication, video stream acquisition, GPU TensorRT
worker IPC, multi-frame decision filtering, and MQTT halting dispatch.
"""

import os
import sys
import time
import json
import signal
import logging
import argparse
import socket
import subprocess
from pathlib import Path

# Add current folder and api folder to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "api"))

from api.bambu_auth import BambuAuth
from api.video_stream import A1VideoStream
from api.mqtt_controller import A1MQTTController
from api.ipc_client import InferenceClient, SOCKET_PATH
from api.decision_engine import DecisionEngine
from api.snapshot import SnapshotManager

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Main")


class SpaghettiDetectionApp:
    def __init__(self, config_path: str, dry_run_override: bool = None):
        self.config_path = config_path
        self.config = self._load_config()

        # Apply overrides
        if dry_run_override is not None:
            self.config.setdefault("safety", {})["dry_run"] = dry_run_override

        self.worker_process = None
        self.running = False

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at '{self.config_path}'. Please copy config.json.example to config.json."
            )
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _ensure_worker_running(self):
        """Starts the JetPack Python 3.6 GPU Inference Worker if socket is not active."""
        det_cfg = self.config.get("detection", {})
        backend = det_cfg.get("backend", "torch")
        if backend not in ("torch", "native_cuda"):
            raise ValueError("Unsupported inference backend: {}".format(backend))
        if os.path.exists(SOCKET_PATH):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as test_sock:
                    test_sock.settimeout(1.0)
                    test_sock.connect(SOCKET_PATH)
                logger.info(f"Existing inference socket active at {SOCKET_PATH}. Reusing worker.")
                return
            except (socket.error, ConnectionRefusedError):
                # Only the worker may remove a stale socket after checking its
                # type, listener status, and inode. Never unlink arbitrary paths.
                logger.info(f"Existing socket unavailable at {SOCKET_PATH}; worker will check it safely.")

        jetpack_python = BASE_DIR / "inference" / ".venv-jetpack" / "bin" / "python"
        worker_script = BASE_DIR / "inference" / "worker_server.py"

        if not jetpack_python.exists():
            # Fallback to system jetpack-python3.6
            jetpack_python = "/usr/local/bin/jetpack-python3.6"

        engine_file = BASE_DIR / det_cfg.get("model_engine", "models/obico_spaghetti.engine")
        model_type = det_cfg.get("model_type", "obico")
        conf_thresh = det_cfg.get("confidence_threshold", 0.35)

        if not engine_file.exists():
            raise FileNotFoundError(
                f"TensorRT engine not found at {engine_file}. Please compile with inference/build_engine.sh first."
            )

        cmd = [
            str(jetpack_python),
            str(worker_script),
            "--engine", str(engine_file),
            "--type", model_type,
            "--conf", str(conf_thresh),
            "--backend", backend
        ]

        logger.info(f"Spawning GPU Inference Worker process: {' '.join(cmd)}")
        env = os.environ.copy()
        env["OPENBLAS_CORETYPE"] = "ARMV8"

        self.worker_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid
        )

        # Loading the engine and prewarming JPEG/CUDA/NMS can exceed 60 seconds
        # under memory pressure. This is separate from the per-frame IPC timeout.
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self.worker_process.poll() is not None:
                raise RuntimeError("Inference worker exited during startup (code {}).".format(
                    self.worker_process.returncode))
            if os.path.exists(SOCKET_PATH):
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ready_sock:
                        ready_sock.settimeout(0.5)
                        ready_sock.connect(SOCKET_PATH)
                    logger.info("Inference worker ready after full pipeline warmup!")
                    return
                except OSError:
                    pass
            time.sleep(0.5)

        self.stop()
        raise TimeoutError("Inference worker failed to become ready within 120 seconds.")

    def run(self):
        self.running = True

        # Handle signals for graceful shutdown
        def sig_handler(sig, frame):
            logger.info(f"Received signal {sig}. Initiating graceful shutdown...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        # 1. Start GPU Inference Worker
        self._ensure_worker_running()

        # 2. Authenticate and resolve credentials
        printer_cfg = self.config.get("printer", {})
        target_serial = printer_cfg.get("target_serial", "03900D6A1521014")
        printer_ip = printer_cfg.get("printer_ip", "192.168.3.231")
        access_code = printer_cfg.get("access_code", "")

        cloud_cfg = self.config.get("bambu_cloud", {})
        account = cloud_cfg.get("account")
        password = cloud_cfg.get("password")
        region = cloud_cfg.get("region", "china")

        if not access_code:
            auth = BambuAuth(region=region)
            creds = None
            if auth.token:
                try:
                    logger.info("Using cached Bambu Cloud token...")
                    creds = auth.get_a1_credentials(target_serial=target_serial)
                except Exception as e:
                    logger.warning(f"Cached token invalid ({e}). Requesting login...")
                    creds = None

            if not creds:
                if not account or not password:
                    raise ValueError("Must provide either 'access_code' or Bambu Cloud 'account'/'password' in config.json")
                auth.login(account, password)
                creds = auth.get_a1_credentials(target_serial=target_serial)

            access_code = creds["access_code"]
            logger.info(f"Retrieved Access Code for A1 from Cloud: {access_code}")

        # 3. Initialize components
        det_cfg = self.config.get("detection", {})
        safety_cfg = self.config.get("safety", {})
        storage_cfg = self.config.get("storage", {})

        decision_engine = DecisionEngine(
            consecutive_required=det_cfg.get("consecutive_hits", 6),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.35),
            target_classes=det_cfg.get("target_classes", ["spaghetti", "warping"]),
            dry_run=safety_cfg.get("dry_run", True),
            action=safety_cfg.get("action", "pause"),
            cooldown_seconds=safety_cfg.get("cooldown_seconds", 180)
        )

        snapshot_mgr = SnapshotManager(
            output_dir=str(BASE_DIR / storage_cfg.get("snapshots_dir", "snapshots"))
        )

        # 4. Start MQTT Controller
        mqtt_controller = A1MQTTController(
            printer_ip=printer_ip,
            access_code=access_code,
            dev_id=target_serial
        )
        mqtt_controller.start()

        # 5. Connect IPC Inference Client
        ipc_client = InferenceClient()
        ipc_client.connect()

        # 6. Connect Video Stream
        video_stream = A1VideoStream(
            printer_ip=printer_ip,
            access_code=access_code
        )

        print("\n" + "=" * 65)
        print(" [READY] Bambu Lab A1 Spaghetti Failure Detection Active")
        print(f"  Target Printer : {target_serial} ({printer_ip})")
        print(f"  Dry-Run Mode   : {'ENABLED (Monitoring Only)' if decision_engine.dry_run else 'DISABLED (Active Print Halting)'}")
        print(f"  Safety Action  : {decision_engine.action.upper()}")
        print(f"  Filter Hits    : {decision_engine.consecutive_required} consecutive frames")
        print("=" * 65 + "\n")

        # 7. Main Monitoring Loop
        frame_idx = 0
        try:
            for jpeg_frame in video_stream.stream_frames():
                if not self.running:
                    break

                frame_idx += 1
                current_state = mqtt_controller.gcode_state

                # Send frame to GPU TensorRT Worker
                t0 = time.time()
                try:
                    detections = ipc_client.detect(jpeg_frame)
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    decision_engine.invalidate_frame()
                    continue
                dt_ms = (time.time() - t0) * 1000.0

                # Decision state machine evaluation
                eval_res = decision_engine.evaluate_frame(detections, current_state)

                # Console telemetry line
                hits = eval_res["consecutive_hits"]
                req = decision_engine.consecutive_required
                conf = eval_res["highest_confidence"]
                status_symbol = "🔴" if hits > 0 else "🟢"
                print(
                    f"\r{status_symbol} [Frame #{frame_idx}] [{dt_ms:.1f}ms] [STATE: {current_state}] [Hits: {hits}/{req}] [Max Conf: {conf:.2f}] [Dets: {len(detections)}]   ",
                    end="",
                    flush=True
                )

                # Handle confirmed failure
                if eval_res["triggered"]:
                    print("\n")
                    logger.critical("🚨 FAILURE TRIGGER ACTIVATED!")

                    # Save evidence snapshot
                    snap_path = snapshot_mgr.save_failure_evidence(
                        jpeg_bytes=jpeg_frame,
                        detections=eval_res["matching_detections"],
                        consecutive_hits=hits,
                        printer_info={
                            "serial": target_serial,
                            "ip": printer_ip,
                            "layer": mqtt_controller.current_layer,
                            "progress": mqtt_controller.progress_percent
                        }
                    )
                    logger.critical(f"Annotated forensic snapshot saved to: {snap_path}")

                    # Dispatch printer control
                    if not decision_engine.dry_run:
                        if decision_engine.action == "stop":
                            logger.critical("Sending STOP command to printer...")
                            mqtt_controller.stop_print()
                        else:
                            logger.critical("Sending PAUSE command to printer...")
                            mqtt_controller.pause_print()
                    else:
                        logger.warning("Dry-Run enabled: Skipping actual printer halt command.")

        finally:
            video_stream.disconnect()
            mqtt_controller.stop()
            ipc_client.close()
            self.stop()

    def stop(self):
        self.running = False
        if self.worker_process:
            try:
                os.killpg(os.getpgid(self.worker_process.pid), signal.SIGTERM)
            except Exception:
                pass
            self.worker_process = None


def main():
    parser = argparse.ArgumentParser(description="Bambu Lab A1 Spaghetti Failure Detector")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Force enable Dry-Run mode")
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run", help="Disable Dry-Run (real halting)")
    args = parser.parse_args()

    app = SpaghettiDetectionApp(
        config_path=args.config,
        dry_run_override=args.dry_run
    )
    app.run()


if __name__ == "__main__":
    main()
