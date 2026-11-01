#!/usr/bin/env python3
"""
Comprehensive Benchmark & Evaluation of New 3D Print Failure Detection Models
=============================================================================
Evaluates:
  1. Bambu AI Monitor (YOLOv8n-Bambu, 640x640)
  2. Failure Monitor (YOLOv8n 3-class, 640x640 & 416x416)
  3. YOLO11s Spaghetti 93% (YOLO11s, 640x640 & 416x416)
  4. YOLO11s 3-Class (spaghetti, stringing, zits; 640x640)
  5. PrintGuard (ShuffleNetV2 Prototype Classifier, 224x224)
  6. YOLOv5s Failure Detector (Baseline Winner, 416x416)
  7. Obico Spaghetti Detective (Baseline Reference, 416x416)
"""

import os
import sys
import glob
import time
import json
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent

# Set cuDNN path for onnxruntime-gpu
cudnn_lib = ROOT_DIR / ".venv" / "lib" / "python3.11" / "site-packages" / "nvidia" / "cudnn" / "lib"
if cudnn_lib.exists():
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"


class ModelEvaluator:
    def __init__(self, models_dir: Path, dataset_dir: Path, conf_thresh: float = 0.30):
        self.models_dir = models_dir
        self.dataset_dir = dataset_dir
        self.conf_thresh = conf_thresh
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print("[INFO] Loading all candidate models into ONNX Runtime (CUDA)...")

        # 1. Bambu AI Monitor (YOLOv8n 640x640)
        p_bambu = models_dir / "bambu_ai_monitor_640.onnx"
        self.sess_bambu = ort.InferenceSession(str(p_bambu), providers=self.providers)
        print("  - Loaded Bambu AI Monitor (YOLOv8n 640x640)")

        # 2. Failure Monitor YOLOv8n (640x640 & 416x416)
        p_v8n_640 = models_dir / "failure_monitor_v8n_640.onnx"
        self.sess_v8n_640 = ort.InferenceSession(str(p_v8n_640), providers=self.providers)
        p_v8n_416 = models_dir / "failure_monitor_v8n_416.onnx"
        self.sess_v8n_416 = ort.InferenceSession(str(p_v8n_416), providers=self.providers)
        print("  - Loaded Failure Monitor YOLOv8n (640 & 416)")

        # 3. YOLO11s Spaghetti 93% (640x640 & 416x416)
        p_y11_640 = models_dir / "yolo11s_spaghetti_93_640.onnx"
        self.sess_y11_640 = ort.InferenceSession(str(p_y11_640), providers=self.providers)
        p_y11_416 = models_dir / "yolo11s_spaghetti_93_416.onnx"
        self.sess_y11_416 = ort.InferenceSession(str(p_y11_416), providers=self.providers)
        print("  - Loaded YOLO11s Spaghetti 93% (640 & 416)")

        # 4. YOLO11s 3-Class (640x640)
        p_y11_3c = models_dir / "yolo11s_spaghetti_3class_640.onnx"
        self.sess_y11_3c = ort.InferenceSession(str(p_y11_3c), providers=self.providers)
        print("  - Loaded YOLO11s 3-Class (640x640)")

        # 5. PrintGuard (ShuffleNetV2 224x224)
        p_pg = models_dir / "printguard_encoder_224.onnx"
        self.sess_pg = ort.InferenceSession(str(p_pg), providers=self.providers)
        with open(models_dir / "printguard_prototypes.json") as f:
            raw_proto = json.load(f)
            proto_dict = raw_proto.get("prototypes", raw_proto)
            self.pg_prototypes = {k: np.asarray(v, dtype=np.float32) for k, v in proto_dict.items()}
        with open(models_dir / "printguard_metadata.json") as f:
            meta = json.load(f)
            self.pg_mean = tuple(float(x) for x in meta["preprocessing"]["normalise_mean"])
            self.pg_std = tuple(float(x) for x in meta["preprocessing"]["normalise_std"])
        print("  - Loaded PrintGuard (ShuffleNetV2 Prototypes)")

        # 6. YOLOv5s Failure (Baseline Champion 416x416)
        p_v5 = models_dir / "yolov5_failure_416.onnx"
        self.sess_v5 = ort.InferenceSession(str(p_v5), providers=self.providers)
        print("  - Loaded YOLOv5s Failure Baseline (416x416)")

        # 7. Obico (Darknet YOLOv3/v4 416x416)
        p_obico = models_dir / "obico_spaghetti.onnx"
        self.sess_obico = ort.InferenceSession(str(p_obico), providers=self.providers)
        print("  - Loaded Obico Baseline (416x416)")

    def letterbox(self, img_bgr, target_size=(640, 640)):
        h, w = img_bgr.shape[:2]
        th, tw = target_size
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        px, py = (tw - nw) // 2, (th - nh) // 2
        canvas[py:py+nh, px:px+nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...]

    def predict_yolo_v8_11(self, sess, tensor, target_indices=(4,)):
        """Evaluates YOLOv8/YOLO11 output: [1, 4 + C, N]"""
        t0 = time.time()
        out = sess.run(None, {"images": tensor})[0].squeeze()  # [4 + C, N]
        lat = (time.time() - t0) * 1000.0

        # out shape: [4+C, N]
        # Slice target class rows
        if isinstance(target_indices, int):
            scores = out[target_indices, :]
        else:
            scores = np.max(out[list(target_indices), :], axis=0)

        max_c = float(np.max(scores)) if len(scores) > 0 else 0.0
        det = bool(max_c >= self.conf_thresh)
        return det, max_c, lat

    def predict_printguard(self, img_bgr):
        """Preprocesses frame, extracts embedding, and calculates Euclidean distance to prototypes."""
        t0 = time.time()
        # RGB PIL image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocessing according to PrintGuard spec
        scale = 256.0 / min(pil_img.size)
        w, h = pil_img.size
        # limit ratio
        w_lim, h_lim = min(w, min(pil_img.size) * 4), min(h, min(pil_img.size) * 4)
        l, t = (w - w_lim) // 2, (h - h_lim) // 2
        pil_img = pil_img.resize((round(w_lim * scale), round(h_lim * scale)), Image.Resampling.BILINEAR, box=(l, t, l + w_lim, t + h_lim))
        
        l, t = (pil_img.width - 224) // 2, (pil_img.height - 224) // 2
        pil_img = pil_img.convert("L").crop((l, t, l + 224, t + 224))
        grey = np.asarray(pil_img, dtype=np.float32) / 255.0
        chans = np.stack([(grey - m) / s for m, s in zip(self.pg_mean, self.pg_std)], axis=0)
        tensor = chans[np.newaxis, ...].astype(np.float32)

        out = self.sess_pg.run(None, {"input": tensor})[0].squeeze()  # [1024]
        lat = (time.time() - t0) * 1000.0

        # Classify by nearest prototype
        d_succ = float(np.linalg.norm(out - self.pg_prototypes["success"]))
        d_fail = float(np.linalg.norm(out - self.pg_prototypes["failure"]))
        
        # Failure detected if d_fail < d_succ
        is_fail = bool(d_fail < d_succ)
        margin = d_succ - d_fail  # positive if failure
        # Normalize pseudo-confidence in [0, 1] using sigmoid of margin
        pseudo_conf = 1.0 / (1.0 + math.exp(-margin * 2.0))
        return is_fail, pseudo_conf, lat

    def predict_yolov5(self, tensor):
        """YOLOv5s Failure (error + spaghetti)"""
        t0 = time.time()
        out = self.sess_v5.run(None, {"images": tensor})[0].squeeze()
        lat = (time.time() - t0) * 1000.0
        obj = out[:, 4]
        # class 0: error, class 3: spaghetti
        scores = obj * np.maximum(out[:, 5], out[:, 8])
        max_c = float(np.max(scores)) if len(scores) > 0 else 0.0
        det = bool(max_c >= self.conf_thresh)
        return det, max_c, lat

    def predict_obico(self, tensor):
        """Obico Darknet YOLO"""
        t0 = time.time()
        boxes, confs = self.sess_obico.run(None, {"input": tensor})
        lat = (time.time() - t0) * 1000.0
        c = confs.squeeze()
        max_c = float(np.max(c)) if len(c) > 0 else 0.0
        det = bool(max_c >= self.conf_thresh)
        return det, max_c, lat

    def run_benchmark(self, split: str = "test"):
        img_dir = self.dataset_dir / "images" / split
        lbl_dir = self.dataset_dir / "labels" / split
        img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))
        print(f"\n================================================================================")
        print(f" BENCHMARKING ON '{split.upper()}' SET ({len(img_paths)} IMAGES) | CONF THRESHOLD = {self.conf_thresh}")
        print(f"================================================================================")

        models = [
            "YOLOv5s (Baseline)",
            "Bambu AI Monitor (v8n-640)",
            "Failure Monitor (v8n-640)",
            "Failure Monitor (v8n-416)",
            "YOLO11s Spaghetti 93% (640)",
            "YOLO11s Spaghetti 93% (416)",
            "YOLO11s 3-Class (640)",
            "PrintGuard (ShuffleNet 224)",
            "Obico (Darknet-416)",
        ]

        stats = {m: {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "latencies": [], "confs": []} for m in models}

        for idx, img_p in enumerate(img_paths, 1):
            stem = Path(img_p).stem
            lbl_p = lbl_dir / f"{stem}.txt"

            has_gt_failure = False
            if lbl_p.exists():
                lines = [l.strip() for l in lbl_p.read_text().splitlines() if l.strip()]
                for l in lines:
                    c = int(l.split()[0])
                    if c == 1:  # Spaghetti failure
                        has_gt_failure = True
                        break

            img = cv2.imread(img_p)
            if img is None:
                continue

            # Preprocessed tensors
            t416 = self.letterbox(img, (416, 416))
            t640 = self.letterbox(img, (640, 640))

            # Inferences
            # 1. YOLOv5s
            d, c, lat = self.predict_yolov5(t416)
            self._record(stats["YOLOv5s (Baseline)"], d, has_gt_failure, c, lat)

            # 2. Bambu AI Monitor (v8n-640) - class 0 is Spaghetti (row 4)
            d, c, lat = self.predict_yolo_v8_11(self.sess_bambu, t640, target_indices=[4])
            self._record(stats["Bambu AI Monitor (v8n-640)"], d, has_gt_failure, c, lat)

            # 3. Failure Monitor (v8n-640) - classes 0, 1, 2 are spaghetti, stringing, warping (rows 4, 5, 6)
            d, c, lat = self.predict_yolo_v8_11(self.sess_v8n_640, t640, target_indices=[4, 5, 6])
            self._record(stats["Failure Monitor (v8n-640)"], d, has_gt_failure, c, lat)

            # 4. Failure Monitor (v8n-416)
            d, c, lat = self.predict_yolo_v8_11(self.sess_v8n_416, t416, target_indices=[4, 5, 6])
            self._record(stats["Failure Monitor (v8n-416)"], d, has_gt_failure, c, lat)

            # 5. YOLO11s 93% (640) - class 0 is spaghetti (row 4)
            d, c, lat = self.predict_yolo_v8_11(self.sess_y11_640, t640, target_indices=[4])
            self._record(stats["YOLO11s Spaghetti 93% (640)"], d, has_gt_failure, c, lat)

            # 6. YOLO11s 93% (416)
            d, c, lat = self.predict_yolo_v8_11(self.sess_y11_416, t416, target_indices=[4])
            self._record(stats["YOLO11s Spaghetti 93% (416)"], d, has_gt_failure, c, lat)

            # 7. YOLO11s 3-Class (640) - classes 0, 1, 2 (rows 4, 5, 6)
            d, c, lat = self.predict_yolo_v8_11(self.sess_y11_3c, t640, target_indices=[4, 5, 6])
            self._record(stats["YOLO11s 3-Class (640)"], d, has_gt_failure, c, lat)

            # 8. PrintGuard
            d, c, lat = self.predict_printguard(img)
            self._record(stats["PrintGuard (ShuffleNet 224)"], d, has_gt_failure, c, lat)

            # 9. Obico
            d, c, lat = self.predict_obico(t416)
            self._record(stats["Obico (Darknet-416)"], d, has_gt_failure, c, lat)

            if idx % 50 == 0 or idx == len(img_paths):
                print(f"  Processed {idx}/{len(img_paths)} frames...")

        # Print Leaderboard
        print("\n" + "=" * 105)
        print(f" MODEL BENCHMARK LEADERBOARD ({split.upper()} SET - {len(img_paths)} IMAGES)")
        print("=" * 105)
        header = f"{'Model':<30} | {'TP':>4} | {'FP':>4} | {'TN':>4} | {'FN':>4} | {'Prec':>7} | {'Recall':>7} | {'F1-Score':>8} | {'Latency':>8}"
        print(header)
        print("-" * 105)

        summary = {}
        # Sort by F1 score descending
        sorted_models = sorted(
            models,
            key=lambda m: self._calc_f1(stats[m]["TP"], stats[m]["FP"], stats[m]["FN"]),
            reverse=True
        )

        for m in sorted_models:
            s = stats[m]
            tp, fp, tn, fn = s["TP"], s["FP"], s["TN"], s["FN"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            avg_lat = np.mean(s["latencies"]) if s["latencies"] else 0.0

            summary[m] = {
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "Avg_Latency_ms": round(avg_lat, 2)
            }

            print(f"{m:<30} | {tp:4d} | {fp:4d} | {tn:4d} | {fn:4d} | {prec:7.2%} | {rec:7.2%} | {f1:8.4f} | {avg_lat:6.2f} ms")

        print("=" * 105)
        return summary

    @staticmethod
    def _calc_f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    @staticmethod
    def _record(stat_dict, detected, has_gt, conf, lat):
        stat_dict["latencies"].append(lat)
        stat_dict["confs"].append(conf)
        if detected and has_gt:
            stat_dict["TP"] += 1
        elif detected and not has_gt:
            stat_dict["FP"] += 1
        elif not detected and not has_gt:
            stat_dict["TN"] += 1
        else:
            stat_dict["FN"] += 1


def main():
    models_dir = ROOT_DIR / "models"
    dataset_dir = ROOT_DIR / "datasets"

    evaluator = ModelEvaluator(models_dir, dataset_dir, conf_thresh=0.30)
    
    # 1. Evaluate on test set
    test_results = evaluator.run_benchmark("test")

    # 2. Save results
    out_file = dataset_dir / "eval_results" / "new_models_benchmark_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n[SUCCESS] Benchmark report saved to: {out_file}")


if __name__ == "__main__":
    main()
