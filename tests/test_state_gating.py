"""
Unit tests for State-Gated GPU Inference.
Verifies that GPU inference is completely bypassed when the printer is not in RUNNING state (e.g. IDLE, PAUSE).
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main


def _fake_jpeg():
    return b"\xff\xd8" + b"\x00" * 32 + b"\xff\xd9"


def test_gpu_inference_bypassed_when_idle(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "printer": {"access_code": "fake", "target_serial": "039TEST"},
        "storage": {"snapshots_dir": str(tmp_path / "photos")},
        "safety": {"dry_run": True}
    }))

    app = main.SpaghettiDetectionApp(str(config))
    app.running = True

    detect_calls = []

    class MockIPC:
        def connect(self): pass
        def close(self): pass
        def detect(self, payload):
            detect_calls.append(payload)
            return []

    monkeypatch.setattr(app, "_ensure_worker_running", lambda: None)
    monkeypatch.setattr(main, "InferenceClient", MockIPC)
    monkeypatch.setattr(main, "A1MQTTController", lambda **kw: SimpleNamespace(
        start=lambda: None, stop=lambda: None, gcode_state="IDLE",
        current_layer=0, progress_percent=0, subtask_name=""
    ))
    monkeypatch.setattr(main, "A1VideoStream", lambda **kw: SimpleNamespace(
        stream_frames=lambda: iter([_fake_jpeg(), _fake_jpeg(), _fake_jpeg()]),
        disconnect=lambda: None
    ))

    app._run_monitoring()

    # When state is IDLE, GPU detect must NEVER be called!
    assert len(detect_calls) == 0, f"Expected 0 GPU inference calls when IDLE, got {len(detect_calls)}"


def test_gpu_inference_invoked_when_running(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "printer": {"access_code": "fake", "target_serial": "039TEST"},
        "storage": {"snapshots_dir": str(tmp_path / "photos")},
        "safety": {"dry_run": True}
    }))

    app = main.SpaghettiDetectionApp(str(config))
    app.running = True

    detect_calls = []

    class MockIPC:
        def connect(self): pass
        def close(self): pass
        def detect(self, payload):
            detect_calls.append(payload)
            return []

    monkeypatch.setattr(app, "_ensure_worker_running", lambda: None)
    monkeypatch.setattr(main, "InferenceClient", MockIPC)
    monkeypatch.setattr(main, "A1MQTTController", lambda **kw: SimpleNamespace(
        start=lambda: None, stop=lambda: None, gcode_state="RUNNING",
        current_layer=1, progress_percent=10, subtask_name="print_job"
    ))
    monkeypatch.setattr(main, "A1VideoStream", lambda **kw: SimpleNamespace(
        stream_frames=lambda: iter([_fake_jpeg(), _fake_jpeg()]),
        disconnect=lambda: None
    ))

    app._run_monitoring()

    # When state is RUNNING, GPU detect must be called for each frame
    assert len(detect_calls) == 2, f"Expected 2 GPU inference calls when RUNNING, got {len(detect_calls)}"
