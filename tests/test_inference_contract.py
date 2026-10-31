"""Layout and warmup regressions. CUDA/TensorRT cases run on Jetson only."""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

try:
    import cv2
    import numpy as np
    import torch
    from gpu_preprocess import GPUPreprocessor
    HAS_IMAGE_RUNTIME = True
except ImportError:
    HAS_IMAGE_RUNTIME = False

HAS_TRT = HAS_IMAGE_RUNTIME and importlib.util.find_spec("tensorrt") is not None


@unittest.skipUnless(HAS_IMAGE_RUNTIME, "Requires the image processing runtime")
class TestPreprocessingLayout(unittest.TestCase):
    def test_rgb_planes_are_contiguous_for_linear_binding(self):
        # Lossless encoding keeps the independently specified colors exact.
        bgr = np.array([[[0, 0, 255], [0, 255, 0]],
                        [[255, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        tensor, _ = GPUPreprocessor((2, 2), device="cpu").decode_and_preprocess(encoded.tobytes())
        self.assertTrue(tensor.is_contiguous(), "TensorRT reads linear NCHW, not strided HWC storage")
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(list(tensor.shape), [1, 3, 2, 2])
        self.assertEqual(tensor.reshape(-1).tolist(),
                         [1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1.])


@unittest.skipUnless(HAS_TRT, "TensorRT worker tests require the Jetson runtime")
class TestFullPipelineWarmup(unittest.TestCase):
    def test_warmup_exercises_jpeg_and_detection_postprocessing(self):
        import worker_server
        from postprocess import GPUPostProcessor

        warmup = getattr(worker_server, "warmup_pipeline", None)
        self.assertIsNotNone(warmup, "Worker needs full image/NMS warmup before listening")
        # Only the external model is replaced; real decode, preprocessing and
        # postprocessing execute on CPU, including nonempty NMS results.
        class FixedDetector:
            def __init__(self, model_type):
                self.model_type = model_type
                self.class_names = ["spaghetti"] if model_type == "obico" else ["spaghetti", "warping"]

            def infer(self, tensor):
                # Real models may have no candidates during warmup. The worker
                # must still prewarm the nonempty NMS/coordinate path.
                if self.model_type != "obico":
                    return [torch.zeros((1, 6, 1))]
                return [torch.tensor([[[[0., 0., 1., 1.]]]]), torch.zeros((1, 1, 1))]

        class ObservedPreprocessor(GPUPreprocessor):
            def __init__(self):
                super().__init__((32, 32), device="cpu")
                self.jpeg_seen = False
                self.resolutions = set()

            def decode_and_preprocess(self, payload):
                self.jpeg_seen = payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")
                tensor, meta = super().decode_and_preprocess(payload)
                self.resolutions.add((meta["orig_w"], meta["orig_h"]))
                return tensor, meta

        class ObservedPostprocessor(GPUPostProcessor):
            def __init__(self):
                super().__init__(device="cpu")
                self.results = []

            def process_obico(self, outputs, meta, class_names):
                result = super().process_obico(outputs, meta, class_names)
                self.results.extend(result)
                return result

            def process_yolo(self, outputs, meta, class_names):
                result = super().process_yolo(outputs, meta, class_names)
                self.results.extend(result)
                return result

        for model_type in ("obico", "yolov8"):
            with self.subTest(model_type=model_type):
                pre = ObservedPreprocessor()
                post = ObservedPostprocessor()
                warmup(FixedDetector(model_type), pre, post, model_type)
                self.assertTrue(pre.jpeg_seen)
                self.assertTrue({(1280, 720), (1536, 1080)}.issubset(pre.resolutions),
                                "Warmup must cover the actual A1 camera resolution as well as 720p")
                self.assertTrue(post.results, "Warmup must exercise nonempty NMS and box conversion")
                self.assertTrue(all(det["label"] == "spaghetti" for det in post.results))


@unittest.skipUnless(HAS_TRT and torch.cuda.is_available(), "Requires CUDA and TensorRT")
class TestTensorRTInputBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine = Path(os.environ.get("SPAGHETTI_TEST_ENGINE", str(ROOT / "models/obico_spaghetti.engine")))
        if not engine.exists():
            raise unittest.SkipTest("No target-specific TensorRT engine")
        from trt_detector import TensorRTDetector
        cls.detector = TensorRTDetector(str(engine))

    def test_noncontiguous_input_matches_contiguous_reference(self):
        h, w = self.detector.get_input_resolution()
        # Distinct spatial/channel values expose a pointer interpreted as HWC.
        hwc = torch.arange(h * w * 3, dtype=torch.float32, device="cuda").reshape(h, w, 3)
        hwc.remainder_(251).div_(250.)
        nchw = hwc.permute(2, 0, 1).unsqueeze(0)
        self.assertFalse(nchw.is_contiguous())
        reference = self.detector.infer(nchw.contiguous())
        actual = self.detector.infer(nchw)
        for want, got in zip(reference, actual):
            self.assertTrue(torch.allclose(want, got, rtol=1e-4, atol=1e-5),
                            "Strided input must not change engine predictions")

    def test_wrong_shape_rejected_before_enqueue(self):
        h, w = self.detector.get_input_resolution()
        bad = torch.zeros((1, 3, h, w - 1), device="cuda")
        with self.assertRaises(ValueError):
            self.detector.infer(bad)

    def test_wrong_dtype_rejected_before_enqueue(self):
        h, w = self.detector.get_input_resolution()
        bad = torch.zeros((1, 3, h, w), dtype=torch.float64, device="cuda")
        with self.assertRaises(ValueError):
            self.detector.infer(bad)


if __name__ == "__main__":
    unittest.main()
