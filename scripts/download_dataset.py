#!/usr/bin/env python3
"""
3D Print Failure Dataset Downloader
===================================
Parses Ultralytics/Roboflow NDJSON dataset manifest and downloads images
and annotations in parallel into standard YOLO format under `datasets/`.

Usage:
    python scripts/download_dataset.py --split test,val
    python scripts/download_dataset.py --split all
"""

import os
import sys
import json
import time
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_manifest(manifest_path: Path):
    """Parses header and records from ndjson file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        records = []
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return header, records


def download_one(record: dict, output_dir: Path):
    """Downloads one image and writes its YOLO label file."""
    split = record.get("split", "train")
    filename = record.get("file")
    url = record.get("url")
    boxes = record.get("annotations", {}).get("boxes", [])

    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / filename
    stem = Path(filename).stem
    lbl_path = lbl_dir / f"{stem}.txt"

    # Write label file
    lines = []
    for b in boxes:
        # Format: class_id, cx, cy, w, h
        cls_id = int(b[0])
        cx, cy, w, h = b[1], b[2], b[3], b[4]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(lbl_path, "w", encoding="utf-8") as lf:
        lf.write("\n".join(lines) + ("\n" if lines else ""))

    # Download image if not exists
    if img_path.exists() and img_path.stat().st_size > 1000:
        return True, 0

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        with open(img_path, "wb") as f:
            f.write(data)
        return True, len(data)
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download 3D Print Failure Dataset from NDJSON")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT_DIR / "3dprint-2-v8-fail-3dprint-aug-spaghetii-yolov8.ndjson"),
        help="Path to NDJSON manifest"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "datasets"),
        help="Destination directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test,val",
        help="Comma-separated splits to download (e.g. 'test,val' or 'all')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Parallel download threads (default: 24)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Limit max images per split (0 for all)"
    )

    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        return 1

    print("=" * 65)
    print(" 3D Print Failure Dataset Downloader")
    print("=" * 65)
    print(f"Manifest : {manifest_path.name}")
    print(f"Output   : {output_dir}")

    header, records = parse_manifest(manifest_path)
    class_names = header.get("class_names", {})
    print(f"Classes  : {class_names}")
    print(f"Total in manifest: {len(records)} records")

    # Filter splits
    target_splits = [s.strip() for s in args.split.split(",")]
    if "all" in target_splits:
        filtered = records
    else:
        filtered = [r for r in records if r.get("split") in target_splits]

    if args.max_images > 0:
        filtered = filtered[:args.max_images]

    print(f"Selected : {len(filtered)} images matching splits: {args.split}")

    # Write dataset.yaml
    yaml_content = f"""# 3D Print Spaghetti Failure Dataset
path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
"""
    for k, v in sorted(class_names.items(), key=lambda x: int(x[0])):
        yaml_content += f"  {k}: {v}\n"

    with open(output_dir / "dataset.yaml", "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    # Save manifest header/metadata
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(header, mf, indent=2)

    # Download in parallel
    print(f"\n[INFO] Downloading {len(filtered)} images with {args.workers} threads...")
    t0 = time.time()
    downloaded_count = 0
    total_bytes = 0
    failed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, r, output_dir): r for r in filtered}
        for future in as_completed(futures):
            success, res = future.result()
            if success:
                downloaded_count += 1
                total_bytes += res
            else:
                failed_count += 1
            if (downloaded_count + failed_count) % 25 == 0 or (downloaded_count + failed_count) == len(filtered):
                pct = (downloaded_count + failed_count) / len(filtered) * 100
                print(
                    f"\r  Progress: {downloaded_count + failed_count}/{len(filtered)} ({pct:.1f}%) | "
                    f"Bytes: {total_bytes / 1e6:.1f} MB | Failed: {failed_count}",
                    end="",
                    flush=True
                )

    dt = time.time() - t0
    print(f"\n\n[SUCCESS] Dataset download complete in {dt:.1f}s!")
    print(f"  Successfully processed: {downloaded_count} images ({total_bytes / 1e6:.1f} MB)")
    if failed_count:
        print(f"  Failed: {failed_count} images")
    print(f"  Saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
