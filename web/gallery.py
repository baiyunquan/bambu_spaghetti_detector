"""Read existing evidence without decoding or retaining image pixels."""
import json
import math
import re
from pathlib import Path
from typing import Optional, List, Dict, Any


class PhotoStore:
    PAGE_SIZE = 24
    EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    def __init__(self, root):
        self.root = Path(root).resolve()

    @staticmethod
    def _sanitize_task(task: Optional[str]) -> str:
        if not task:
            return "default"
        clean = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]+', '_', str(task).strip()).strip('_')
        return clean if clean else "default"

    def _inside_dir(self, directory: Path, filename: str) -> Path:
        if not filename or Path(filename).name != filename or '\x00' in filename:
            raise FileNotFoundError(filename)
        try:
            path = (directory / filename).resolve(strict=True)
            if not path.is_relative_to(directory) or not path.is_file():
                raise FileNotFoundError(filename)
            return path
        except (OSError, RuntimeError, ValueError) as exc:
            raise FileNotFoundError(filename) from exc

    def list_tasks(self) -> List[Dict[str, Any]]:
        tasks_map = {}
        # Count default category: both root direct files and root/default files
        default_count = 0
        default_max_mtime = 0
        if self.root.is_dir():
            # 1. Root direct files (legacy)
            for f in self.root.iterdir():
                if f.is_file() and f.suffix.lower() in self.EXTENSIONS:
                    default_count += 1
                    try:
                        mt = f.stat().st_mtime
                        if mt > default_max_mtime:
                            default_max_mtime = mt
                    except OSError:
                        pass
            # 2. Subdirectories
            for entry in self.root.iterdir():
                if entry.is_dir() and not entry.name.startswith('.'):
                    count = 0
                    max_mtime = 0
                    try:
                        for f in entry.iterdir():
                            if f.suffix.lower() in self.EXTENSIONS:
                                count += 1
                                try:
                                    mt = f.stat().st_mtime
                                    if mt > max_mtime:
                                        max_mtime = mt
                                except OSError:
                                    pass
                    except OSError:
                        pass
                    if entry.name == 'default':
                        default_count += count
                        if max_mtime > default_max_mtime:
                            default_max_mtime = max_mtime
                    else:
                        tasks_map[entry.name] = {
                            'name': entry.name,
                            'count': count,
                            'last_modified': max_mtime,
                            'is_default': False
                        }
        tasks_map['default'] = {
            'name': 'default',
            'count': default_count,
            'last_modified': default_max_mtime,
            'is_default': True
        }
        result = list(tasks_map.values())
        result.sort(key=lambda t: (not t['is_default'], t['last_modified']), reverse=True)
        return result

    def open_image(self, filename: str, task: Optional[str] = None) -> Path:
        if Path(filename).suffix.lower() not in self.EXTENSIONS:
            raise FileNotFoundError(filename)
        if task and task != "default":
            task_dir = self.root / self._sanitize_task(task)
            return self._inside_dir(task_dir, filename)

        # For default (or unassigned): check default/ then root
        default_dir = self.root / "default"
        if default_dir.is_dir():
            try:
                return self._inside_dir(default_dir, filename)
            except FileNotFoundError:
                pass
        try:
            return self._inside_dir(self.root, filename)
        except FileNotFoundError:
            pass

        # Search across all task subdirectories if not found in default/root
        if self.root.is_dir():
            for sub in self.root.iterdir():
                if sub.is_dir() and not sub.name.startswith('.') and sub.name != "default":
                    try:
                        return self._inside_dir(sub, filename)
                    except FileNotFoundError:
                        continue
        raise FileNotFoundError(filename)

    def list_page(self, page: int = 1, task: Optional[str] = None):
        tasks = self.list_tasks()
        task_clean = self._sanitize_task(task)

        candidates = []
        seen = set()
        if task_clean == "default":
            # Include files from default/ subfolder
            default_dir = self.root / "default"
            if default_dir.is_dir():
                for entry in default_dir.iterdir():
                    if entry.suffix.lower() in self.EXTENSIONS:
                        try:
                            path = self._inside_dir(default_dir, entry.name)
                            candidates.append((path.stat().st_mtime_ns, entry.name))
                            seen.add(entry.name)
                        except OSError:
                            continue
            # Also include legacy files directly in root
            if self.root.is_dir():
                for entry in self.root.iterdir():
                    if entry.is_file() and entry.suffix.lower() in self.EXTENSIONS and entry.name not in seen:
                        try:
                            path = self._inside_dir(self.root, entry.name)
                            candidates.append((path.stat().st_mtime_ns, entry.name))
                        except OSError:
                            continue
        else:
            task_dir = self.root / task_clean
            if not task_dir.is_dir():
                raise FileNotFoundError(f"Task category '{task}' not found")
            for entry in task_dir.iterdir():
                if entry.suffix.lower() in self.EXTENSIONS:
                    try:
                        path = self._inside_dir(task_dir, entry.name)
                        candidates.append((path.stat().st_mtime_ns, entry.name))
                    except OSError:
                        continue

        candidates.sort(reverse=True)
        pages = max(1, (len(candidates) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        if page < 1 or (len(candidates) > 0 and page > pages) or (len(candidates) == 0 and page > 1):
            raise FileNotFoundError('Page not found')
        start = (page - 1) * self.PAGE_SIZE
        photos = [name for _, name in candidates[start:start+self.PAGE_SIZE]]
        return {
            'photos': photos,
            'page': page,
            'pages': pages,
            'total': len(candidates),
            'task': task_clean,
            'tasks': tasks
        }

    def detail(self, filename: str, task: Optional[str] = None):
        img_path = self.open_image(filename, task=task)
        task_dir = img_path.parent
        metadata = None
        try:
            sidecar_path = self._inside_dir(task_dir, Path(filename).with_suffix('.json').name)
            value = json.loads(sidecar_path.read_text(encoding='utf-8'))
            if not isinstance(value, dict) or not isinstance(value.get('timestamp'), str):
                raise ValueError('Invalid metadata')
            hits = value.get('consecutive_hits')
            detections = value.get('detections')
            if type(hits) is not int or hits < 0 or not isinstance(detections, list):
                raise ValueError('Invalid metadata')
            rows = []
            for detection in detections:
                if not isinstance(detection, dict) or not isinstance(detection.get('label'), str):
                    raise ValueError('Invalid detection')
                confidence, bbox = detection.get('confidence'), detection.get('bbox')
                if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                        or not 0 <= confidence <= 1 or not isinstance(bbox, list) or len(bbox) != 4
                        or any(type(n) not in (int, float) or not math.isfinite(n) for n in bbox)):
                    raise ValueError('Invalid detection')
                rows.append({'label': detection['label'], 'confidence': confidence, 'bbox': bbox,
                             'class_id': detection.get('class_id')})
            metadata = {
                'timestamp': value['timestamp'],
                'consecutive_hits': hits,
                'detections': rows,
                'task_name': value.get('task_name', task_dir.name)
            }
            metadata['inference_error'] = bool(value.get('inference_error'))
            metadata['capture_kind'] = value.get('capture_kind') if value.get('capture_kind') in ('print', 'manual', 'failure') else None
            metadata['manual_requested'] = value.get('manual_requested') is True
        except (OSError, ValueError, UnicodeError, RecursionError, OverflowError):
            pass
        return {'filename': filename, 'task': task_dir.name, 'metadata': metadata}

