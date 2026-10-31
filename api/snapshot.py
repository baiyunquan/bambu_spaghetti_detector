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
        """
        Renders bounding boxes onto the frame and saves the evidence snapshot.

        Returns:
            Saved image file path string.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"failure_{timestamp_str}.jpg"
        json_filename = f"failure_{timestamp_str}.json"
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
        header_text = f"[ALERT] SPAGHETTI FAILURE CONFIRMED ({consecutive_hits} consecutive hits) | {timestamp_str}"
        draw.rectangle([10, 10, 10 + len(header_text) * 9 + 20, 36], fill=(0, 0, 0))
        draw.text((15, 15), header_text, fill=(255, 50, 50))

        # Save annotated image
        image.save(str(img_path), "JPEG", quality=90)

        # Save metadata JSON
        metadata = {
            "timestamp": timestamp_str,
            "consecutive_hits": consecutive_hits,
            "detections": detections,
            "printer": printer_info or {}
        }
        with open(str(json_path), "w") as f:
            json.dump(metadata, f, indent=2)

        return str(img_path)
