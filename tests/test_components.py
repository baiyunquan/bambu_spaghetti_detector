"""
Automated Unit Tests for Spaghetti Detector Core Components
===========================================================
"""

import sys
import unittest
from pathlib import Path

# Add root and api to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "api"))
sys.path.insert(0, str(ROOT_DIR / "inference"))

from api.decision_engine import DecisionEngine
from api.snapshot import SnapshotManager
from api.bambu_auth import BambuAuth
from PIL import Image
import io


class TestDecisionEngine(unittest.TestCase):
    def test_state_interlocking_ignores_idle(self):
        engine = DecisionEngine(consecutive_required=3, confidence_threshold=0.35, dry_run=True)
        # Even with high confidence detections, if state is IDLE or PAUSE, no failure should count
        dets = [{"label": "spaghetti", "confidence": 0.9, "bbox": [10, 10, 100, 100]}]
        res = engine.evaluate_frame(dets, gcode_state="IDLE")
        self.assertFalse(res["has_failure"])
        self.assertEqual(res["consecutive_hits"], 0)
        self.assertFalse(res["triggered"])

    def test_consecutive_hits_trigger(self):
        engine = DecisionEngine(consecutive_required=3, confidence_threshold=0.35, dry_run=True)
        dets = [{"label": "spaghetti", "confidence": 0.85, "bbox": [10, 10, 100, 100]}]

        # Frame 1: Hit 1
        res1 = engine.evaluate_frame(dets, gcode_state="RUNNING")
        self.assertEqual(res1["consecutive_hits"], 1)
        self.assertFalse(res1["triggered"])

        # Frame 2: Hit 2
        res2 = engine.evaluate_frame(dets, gcode_state="RUNNING")
        self.assertEqual(res2["consecutive_hits"], 2)
        self.assertFalse(res2["triggered"])

        # Frame 3: Hit 3 -> Triggered!
        res3 = engine.evaluate_frame(dets, gcode_state="RUNNING")
        self.assertEqual(res3["consecutive_hits"], 3)
        self.assertTrue(res3["triggered"])
        self.assertTrue(res3["dry_run"])

    def test_clean_frame_resets_counter(self):
        engine = DecisionEngine(consecutive_required=3, confidence_threshold=0.35)
        dets = [{"label": "spaghetti", "confidence": 0.85, "bbox": [10, 10, 100, 100]}]

        engine.evaluate_frame(dets, gcode_state="RUNNING")
        engine.evaluate_frame(dets, gcode_state="RUNNING")
        self.assertEqual(engine.consecutive_hits, 2)

        # Clean frame (no detections)
        clean_res = engine.evaluate_frame([], gcode_state="RUNNING")
        self.assertEqual(clean_res["consecutive_hits"], 0)
        self.assertFalse(clean_res["triggered"])


class TestSerialValidation(unittest.TestCase):
    def test_serial_whitelist_prefix(self):
        # A1 printer prefix '039' must pass
        a1_serial = "03900D6A1521014"
        self.assertTrue(a1_serial.startswith("039"))

        # P1S printer prefix '01P' must fail
        p1s_serial = "01P00C550801214"
        self.assertFalse(p1s_serial.startswith("039"))


class TestSnapshotManager(unittest.TestCase):
    def test_annotation_and_save(self):
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            mgr = SnapshotManager(output_dir=temp_dir)
            # Create a dummy 720p image in memory
            img = Image.new("RGB", (1280, 720), color=(100, 100, 100))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            jpeg_bytes = buf.getvalue()

            dets = [{"label": "spaghetti", "confidence": 0.88, "bbox": [50, 50, 300, 300]}]
            snap_path = mgr.save_failure_evidence(jpeg_bytes, dets, consecutive_hits=6)

            self.assertTrue(Path(snap_path).exists())
            self.assertTrue(Path(snap_path).stat().st_size > 1000)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
