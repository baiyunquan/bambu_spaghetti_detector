#!/usr/bin/env python3
"""
Model Benchmark & Evaluation Script
===================================
Compares Obico, YOLOv8, and YOLOv5 on the 3D Print Failure test dataset.
Evaluates Precision, Recall, F1 Score, False Positive Rate, and Latency on GPU.

Usage:
    python scripts/evaluate_models.py
"""

import os
import sys
import glob
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent

# Set cuDNN path for onnxruntime-gpu
cudnn_lib = ROOT_DIR / ".venv" / "lib" / "python3.11" / "site-packages" / "nvidia" / "cudnn" / "lib"
if cudnn_lib.exists():
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"


class ModelEvaluator:
    def __init__(self, models_dir: Path, dataset_dir: Path, conf_thresh: float = 0.35):
        self.models_dir = models_dir
        self.dataset_dir = dataset_dir
        self.conf_thresh = conf_thresh

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[INFO] Initializing ONNX Runtime sessions with CUDA...")

        # 1. Obico Model (416x416)
        obico_path = models_dir / "obico_spaghetti.onnx"
        self.sess_obico = ort.InferenceSession(str(obico_path), providers=providers)
        print(f"  Loaded Obico ONNX: {obico_path.name}")

        # 2. YOLOv8 Failure Model (416x416)
        v8_path = models_dir / "yolov8_failure_416.onnx"
        self.sess_v8 = ort.InferenceSession(str(v8_path), providers=providers)
        print(f"  Loaded YOLOv8 ONNX: {v8_path.name}")

        # 3. YOLOv5s Failure Model (416x416)
        v5_path = models_dir / "yolov5_failure_416.onnx"
        self.sess_v5 = ort.InferenceSession(str(v5_path), providers=providers)
        print(f"  Loaded YOLOv5 ONNX: {v5_path.name}")

    def preprocess_image(self, img_bgr, target_size=(416, 416)):
        """Aspect-ratio letterbox resize and normalization."""
        h, w = img_bgr.shape[:2]
        target_h, target_w = target_size
        scale = min(target_w / w, target_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x = (target_w - nw) // 2
        pad_y = (target_h - nh) // 2

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]  # [1, 3, H, W]

        meta = {
            "orig_w": w, "orig_h": h,
            "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
            "target_w": target_w, "target_h": target_h
        }
        return tensor, meta

    def predict_obico(self, tensor, meta):
        """Runs Obico model and returns detected spaghetti boxes & max conf."""
        t0 = time.time()
        boxes, confs = self.sess_obico.run(None, {"input": tensor})
        latency = (time.time() - t0) * 1000.0

        confs = confs.squeeze()
        if confs.ndim == 0:
            confs = np.array([confs])

        mask = confs >= self.conf_thresh
        detected = bool(np.any(mask))
        max_conf = float(np.max(confs)) if len(confs) > 0 else 0.0

        dets = []
        if detected:
            b_filt = boxes.reshape(-1, 4)[mask]
            c_filt = confs[mask]
            for b, c in zip(b_filt, c_filt):
                # Obico boxes are normalized xyxy
                dets.append({
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "conf": float(c),
                    "label": "spaghetti"
                })

        return detected, max_conf, latency, dets

    def predict_yolov8(self, tensor, meta):
        """Runs YOLOv8 model and returns detected spaghetti boxes & max conf."""
        t0 = time.time()
        out = self.sess_v8.run(None, {"images": tensor})[0].squeeze()  # [6, 3549]
        latency = (time.time() - t0) * 1000.0

        preds = out.transpose(1, 0)  # [3549, 6]
        # class 0: spaghetti, class 1: warping
        spag_scores = preds[:, 4]

        mask = spag_scores >= self.conf_thresh
        detected = bool(np.any(mask))
        max_conf = float(np.max(spag_scores)) if len(spag_scores) > 0 else 0.0

        dets = []
        if detected:
            b_filt = preds[mask, :4]
            c_filt = spag_scores[mask]
            target_w, target_h = meta["target_w"], meta["target_h"]
            for b, c in zip(b_filt, c_filt):
                # cx, cy, w, h in 416 space -> normalized xyxy
                cx, cy, bw, bh = b[0], b[1], b[2], b[3]
                x1 = (cx - bw * 0.5) / target_w
                y1 = (cy - bh * 0.5) / target_h
                x2 = (cx + bw * 0.5) / target_w
                y2 = (cy + bh * 0.5) / target_h
                dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(c),
                    "label": "spaghetti"
                })

        return detected, max_conf, latency, dets

    def predict_yolov5(self, tensor, meta):
        """Runs YOLOv5 model and returns detected spaghetti boxes & max conf."""
        t0 = time.time()
        out = self.sess_v5.run(None, {"images": tensor})[0].squeeze()  # [10647, 9]
        latency = (time.time() - t0) * 1000.0

        # out: [cx, cy, w, h, obj_conf, cls0(error), cls1(extrusor), cls2(part), cls3(spaghetti)]
        obj_conf = out[:, 4]
        spag_scores = obj_conf * out[:, 8]

        mask = spag_scores >= self.conf_thresh
        detected = bool(np.any(mask))
        max_conf = float(np.max(spag_scores)) if len(spag_scores) > 0 else 0.0

        dets = []
        if detected:
            b_filt = out[mask, :4]
            c_filt = spag_scores[mask]
            target_w, target_h = meta["target_w"], meta["target_h"]
            for b, c in zip(b_filt, c_filt):
                cx, cy, bw, bh = b[0], b[1], b[2], b[3]
                x1 = (cx - bw * 0.5) / target_w
                y1 = (cy - bh * 0.5) / target_h
                x2 = (cx + bw * 0.5) / target_w
                y2 = (cy + bh * 0.5) / target_h
                dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(c),
                    "label": "spaghetti"
                })

        return detected, max_conf, latency, dets

    def evaluate_test_set(self, split: str = "test"):
        img_dir = self.dataset_dir / "images" / split
        lbl_dir = self.dataset_dir / "labels" / split

        img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))
        print(f"\n[INFO] Evaluating {len(img_paths)} images from '{split}' split...")

        stats = {
            "Obico": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "latencies": [], "confs": []},
            "YOLOv8": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "latencies": [], "confs": []},
            "YOLOv5s": {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "latencies": [], "confs": []},
        }

        gt_spaghetti_count = 0
        gt_normal_count = 0

        # Output sample dir
        sample_dir = self.dataset_dir / "eval_results"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved_samples = 0

        for idx, img_p in enumerate(img_paths, 1):
            stem = Path(img_p).stem
            lbl_p = lbl_dir / f"{stem}.txt"

            # Parse ground truth
            # Class 1 is Spaghetti
            has_gt_spaghetti = False
            gt_boxes = []
            if lbl_p.exists():
                with open(lbl_p, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id == 1:
                                has_gt_spaghetti = True
                                gt_boxes.append([float(x) for x in parts[1:5]])

            if has_gt_spaghetti:
                gt_spaghetti_count += 1
            else:
                gt_normal_count += 1

            # Read image and preprocess
            img = cv2.imread(img_p)
            if img is None:
                continue

            tensor, meta = self.preprocess_image(img, (416, 416))

            # Run 3 models
            ob_det, ob_conf, ob_lat, ob_boxes = self.predict_obico(tensor, meta)
            v8_det, v8_conf, v8_lat, v8_boxes = self.predict_yolov8(tensor, meta)
            v5_det, v5_conf, v5_lat, v5_boxes = self.predict_yolov5(tensor, meta)

            preds = {
                "Obico": (ob_det, ob_conf, ob_lat, ob_boxes),
                "YOLOv8": (v8_det, v8_conf, v8_lat, v8_boxes),
                "YOLOv5s": (v5_det, v5_conf, v5_lat, v5_boxes),
            }

            for model_name, (is_det, max_c, lat, _) in preds.items():
                stats[model_name]["latencies"].append(lat)
                stats[model_name]["confs"].append(max_c)

                if has_gt_spaghetti:
                    if is_det:
                        stats[model_name]["TP"] += 1
                    else:
                        stats[model_name]["FN"] += 1
                else:
                    if is_det:
                        stats[model_name]["FP"] += 1
                    else:
                        stats[model_name]["TN"] += 1

            # Save sample visualizations for notable cases
            if saved_samples < 8:
                # Save if there is interesting discrepancy or strong detection
                should_save = False
                if has_gt_spaghetti and (ob_det or v8_det or v5_det):
                    should_save = True
                elif not has_gt_spaghetti and (ob_det or v8_det or v5_det):
                    should_save = True  # False alarm case

                if should_save:
                    vis_img = img.copy()
                    ih, iw = img.shape[:2]

                    # Draw GT (Green)
                    for gb in gt_boxes:
                        gcx, gcy, gw, gh = gb
                        gx1, gy1 = int((gcx - gw / 2) * iw), int((gcy - gh / 2) * ih)
                        gx2, gy2 = int((gcx + gw / 2) * iw), int((gcy + gh / 2) * ih)
                        cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                        cv2.putText(vis_img, "GT Spaghetti", (gx1, max(gy1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    # Text overlays with predictions
                    status_str = f"GT: {'SPAGHETTI' if has_gt_spaghetti else 'NORMAL'}"
                    cv2.putText(vis_img, status_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if has_gt_spaghetti else (200, 200, 200), 2)

                    y_off = 50
                    for m_name, (is_d, mc, _, _) in preds.items():
                        col = (0, 0, 255) if is_d else (180, 180, 180)
                        txt = f"{m_name}: {'FAIL' if is_d else 'OK'} (conf: {mc:.2f})"
                        cv2.putText(vis_img, txt, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                        y_off += 22

                    sample_path = sample_dir / f"sample_{saved_samples + 1}_{stem[:15]}.jpg"
                    cv2.imwrite(str(sample_path), vis_img)
                    saved_samples += 1

            if idx % 50 == 0 or idx == len(img_paths):
                print(f"\r  Processed {idx}/{len(img_paths)} images...", end="", flush=True)

        print("\n\n" + "=" * 70)
        print(f" BENCHMARK EVALUATION RESULTS ({split.upper()} SET: {len(img_paths)} images)")
        print(f" Ground Truth: {gt_spaghetti_count} Spaghetti Fails | {gt_normal_count} Normal Prints")
        print("=" * 70)

        results = {}
        for m_name, m_data in stats.items():
            tp = m_data["TP"]
            fp = m_data["FP"]
            tn = m_data["TN"]
            fn = m_data["FN"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False alarm rate on normal prints
            avg_lat = np.mean(m_data["latencies"])

            results[m_name] = {
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
                "FP_Rate": fp_rate,
                "Avg_Latency_ms": avg_lat
            }

            print(f"\n--- {m_name} ---")
            print(f"  Confusion Matrix : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  Precision (查准率): {precision * 100:.2f}%")
            print(f"  Recall    (召回率): {recall * 100:.2f}%")
            print(f"  F1 Score  (综合分): {f1:.4f}")
            print(f"  Accuracy  (准确率): {accuracy * 100:.2f}%")
            print(f"  False Alarm Rate (正常帧误报率): {fp_rate * 100:.2f}% ({fp}/{fp + tn})")
            print(f"  Avg Latency (GPU RTX 5060 Ti)  : {avg_lat:.2f} ms")

        # Save summary report as JSON
        report_path = self.dataset_dir / "eval_results" / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SUCCESS] Detailed benchmark report saved to: {report_path}")
        return results


def main():
    models_dir = ROOT_DIR / "models"
    dataset_dir = ROOT_DIR / "datasets"

    evaluator = ModelEvaluator(models_dir=models_dir, dataset_dir=dataset_dir, conf_thresh=0.35)
    evaluator.evaluate_test_set("test")


if __name__ == "__main__":
    main()
