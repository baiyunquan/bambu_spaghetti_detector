"""
Forensic Snapshot & Evidence Storage Module
===========================================
Annotates video frames with detection bounding boxes, confidence scores,
and timestamps, then saves evidence images to disk.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import io
from uuid import uuid4


class SnapshotManager:
    def __init__(self, output_dir: str = "snapshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_failure_evidence(
        self,
        jpeg_bytes: bytes,
        detections: List[Dict[str, Any]],
        consecutive_hits: int,
        printer_info: Dict[str, Any] = None
    ) -> str:
        return self.save_frame(jpeg_bytes, detections, consecutive_hits, printer_info,
                               capture_kind='failure')

    def save_frame(self, jpeg_bytes, detections, consecutive_hits, printer_info=None,
                   capture_kind='print', manual_requested=False, inference_error=None):
        """Publish one annotated JPEG and sidecar without exposing partial files."""
        if inference_error is not None:
            inference_error = str(inference_error) or 'Inference failed'
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = '{}_{}_{}'.format(capture_kind, timestamp_str, uuid4().hex[:8])
        img_filename = stem + '.jpg'
        json_filename = stem + '.json'
        img_path = self.output_dir / img_filename
        json_path = self.output_dir / json_filename

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
