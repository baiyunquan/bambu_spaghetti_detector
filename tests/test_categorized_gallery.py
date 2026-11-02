import json
import os
import sys
from pathlib import Path
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.gallery import PhotoStore
from api.snapshot import SnapshotManager


def make_dummy_photo(folder: Path, name: str, task_name: str = "default", label: str = "spaghetti"):
    folder.mkdir(parents=True, exist_ok=True)
    img_path = folder / name
    Image.new("RGB", (64, 40), (200, 200, 200)).save(img_path)
    sidecar = img_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "timestamp": "20261102_120000_12345678",
        "task_name": task_name,
        "consecutive_hits": 6,
        "detections": [{"label": label, "class_id": 0, "confidence": 0.95, "bbox": [5, 5, 50, 35]}]
    }))
    return img_path


def test_categorized_storage_and_task_listing(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()

    # 1. Legacy file directly in root
    make_dummy_photo(root, "legacy_01.jpg", task_name="default")

    # 2. Files in default/
    make_dummy_photo(root / "default", "default_01.jpg", task_name="default")
    make_dummy_photo(root / "default", "default_02.jpg", task_name="default")

    # 3. Files in task 'Benchy_3D'
    make_dummy_photo(root / "Benchy_3D", "benchy_01.jpg", task_name="Benchy_3D")
    make_dummy_photo(root / "Benchy_3D", "benchy_02.jpg", task_name="Benchy_3D")

    # 4. Files in task 'Phone_Stand'
    make_dummy_photo(root / "Phone_Stand", "stand_01.jpg", task_name="Phone_Stand")

    store = PhotoStore(root)
    tasks = store.list_tasks()

    task_names = [t["name"] for t in tasks]
    assert "default" in task_names
    assert "Benchy_3D" in task_names
    assert "Phone_Stand" in task_names

    default_task = next(t for t in tasks if t["name"] == "default")
    assert default_task["count"] == 3  # 1 legacy root + 2 in default/

    benchy_task = next(t for t in tasks if t["name"] == "Benchy_3D")
    assert benchy_task["count"] == 2

    # Query page for Benchy
    benchy_page = store.list_page(page=1, task="Benchy_3D")
    assert benchy_page["total"] == 2
    assert "benchy_01.jpg" in benchy_page["photos"]
    assert "benchy_02.jpg" in benchy_page["photos"]
    assert "stand_01.jpg" not in benchy_page["photos"]

    # Query detail
    detail = store.detail("benchy_01.jpg", task="Benchy_3D")
    assert detail["filename"] == "benchy_01.jpg"
    assert detail["task"] == "Benchy_3D"
    assert detail["metadata"]["consecutive_hits"] == 6


def test_snapshot_manager_task_subdirectories(tmp_path):
    root = tmp_path / "snapshots"
    mgr = SnapshotManager(str(root))

    img = Image.new("RGB", (64, 40), (100, 150, 200))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    # Save under specific task
    path1 = Path(mgr.save_failure_evidence(
        jpeg_bytes=jpeg_bytes,
        detections=[{"label": "spaghetti", "confidence": 0.9, "bbox": [10, 10, 50, 30]}],
        consecutive_hits=6,
        printer_info={"subtask_name": "Dragon_Model/Special:Chars"},
        task_name="Dragon_Model/Special:Chars"
    ))
    assert path1.exists()
    assert path1.parent.name == "Dragon_Model_Special_Chars"

    # Save under default when no task given
    path2 = Path(mgr.save_frame(
        jpeg_bytes=jpeg_bytes,
        detections=[],
        consecutive_hits=0,
        capture_kind="manual"
    ))
    assert path2.exists()
    assert path2.parent == root


def test_directory_traversal_rejection(tmp_path):
    root = tmp_path / "snapshots"
    root.mkdir()
    secret = tmp_path / "secret.jpg"
    Image.new("RGB", (32, 32)).save(secret)

    store = PhotoStore(root)
    with pytest.raises(FileNotFoundError):
        store.open_image("../secret.jpg", task="default")

    with pytest.raises(FileNotFoundError):
        store.open_image("secret.jpg", task="../../")
