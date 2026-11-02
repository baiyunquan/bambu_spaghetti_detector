"""
Forensic Snapshot & Evidence Storage Module
===========================================
Annotates video frames with detection bounding boxes, confidence scores,
and timestamps, then saves evidence images to disk.
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import io
from uuid import uuid4


class SnapshotManager:
    def __init__(self, output_dir: str = "snapshots"):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_task_name(task_name: Optional[str]) -> str:
        if not task_name:
            return "default"
        clean = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]+', '_', str(task_name).strip()).strip('_')
        return clean if clean else "default"

    def migrate_legacy_files(self):
        """Migrate any legacy flat files in output_dir into output_dir/default."""
        default_dir = self.output_dir / "default"
        exts = {'.jpg', '.jpeg', '.png', '.json', '.tmp'}
        try:
            legacy_files = [f for f in self.output_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
            if legacy_files:
                default_dir.mkdir(parents=True, exist_ok=True)
                for f in legacy_files:
                    try:
                        dest = default_dir / f.name
                        if not dest.exists():
                            f.rename(dest)
                    except OSError:
                        pass
        except OSError:
            pass

    def save_failure_evidence(
        self,
        jpeg_bytes: bytes,
        detections: List[Dict[str, Any]],
        consecutive_hits: int,
        printer_info: Dict[str, Any] = None,
        task_name: Optional[str] = None
    ) -> str:
        return self.save_frame(jpeg_bytes, detections, consecutive_hits, printer_info,
                                capture_kind='failure', task_name=task_name)

    def save_frame(self, jpeg_bytes, detections, consecutive_hits, printer_info=None,
                   capture_kind='print', manual_requested=False, inference_error=None,
                   task_name: Optional[str] = None):
        """Publish one annotated JPEG and sidecar into task-categorized subdirectory."""
        if inference_error is not None:
            inference_error = str(inference_error) or 'Inference failed'
        category = self.sanitize_task_name(task_name or (printer_info.get("subtask_name") if printer_info else None))
        target_dir = self.output_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = '{}_{}_{}'.format(capture_kind, timestamp_str, uuid4().hex[:8])
        img_filename = stem + '.jpg'
        json_filename = stem + '.json'
        img_path = target_dir / img_filename
        json_path = target_dir / json_filename

        # Load image from JPEG bytes
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Draw red bounding boxes and labels
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            label = det.get("label", "failure")
            conf = det.get("confidence", 0.0)

            x1, y1, x2, y2 = bbox
            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            # Label banner
            text = f"{label.upper()} {conf:.2f}"
            text_y = max(5, y1 - 20)
            draw.rectangle([x1, text_y, x1 + len(text) * 9 + 10, text_y + 18], fill=(255, 0, 0))
            draw.text((x1 + 4, text_y + 2), text, fill=(255, 255, 255))

        # Header watermark
        if capture_kind == 'failure':
            header_text = f"[ALERT] FAILURE CONFIRMED ({consecutive_hits} consecutive hits) | {timestamp_str}"
        else:
            header_text = '{} | {} | {}'.format(capture_kind.upper(),
                'INFERENCE ERROR' if inference_error else '{} detections'.format(len(detections)), timestamp_str)
        draw.rectangle([10, 10, 10 + len(header_text) * 9 + 20, 36], fill=(0, 0, 0))
        draw.text((15, 15), header_text, fill=(255, 50, 50))

        metadata = {
            "timestamp": timestamp_str,
            "task_name": category,
            "consecutive_hits": consecutive_hits,
            "detections": detections,
            "printer": printer_info or {},
            "capture_kind": capture_kind,
            "manual_requested": manual_requested,
            "inference_error": inference_error
        }
        img_tmp = img_path.with_suffix('.jpg.tmp')
        json_tmp = json_path.with_suffix('.json.tmp')
        try:
            image.save(str(img_tmp), "JPEG", quality=90)
            json_tmp.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            json_tmp.replace(json_path)
            img_tmp.replace(img_path)
        except BaseException:
            for path in (img_tmp, json_tmp, json_path):
                path.unlink(missing_ok=True)
            raise
        finally:
            image.close()

        return str(img_path)
