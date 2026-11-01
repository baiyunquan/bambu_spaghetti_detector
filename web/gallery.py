"""Read existing evidence without decoding or retaining image pixels."""
import json
import math
from pathlib import Path


class PhotoStore:
    PAGE_SIZE = 24
    EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    def __init__(self, root):
        self.root = Path(root).resolve()

    def _inside(self, filename):
        if not filename or Path(filename).name != filename or '\x00' in filename:
            raise FileNotFoundError(filename)
        try:
            path = (self.root / filename).resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file():
                raise FileNotFoundError(filename)
            return path
        except (OSError, RuntimeError, ValueError) as exc:
            raise FileNotFoundError(filename) from exc

    def open_image(self, filename):
        path = self._inside(filename)
        if Path(filename).suffix.lower() not in self.EXTENSIONS:
            raise FileNotFoundError(filename)
        return path

    def list_page(self, page):
        candidates = []
        try:
            for entry in self.root.iterdir():
                if entry.suffix.lower() not in self.EXTENSIONS:
                    continue
                try:
                    path = self.open_image(entry.name)
                    candidates.append((path.stat().st_mtime_ns, entry.name))
                except OSError:
                    continue
        except FileNotFoundError:
            pass
        candidates.sort(reverse=True)
        pages = max(1, (len(candidates) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        if page < 1 or page > pages:
            raise FileNotFoundError('Page not found')
        start = (page - 1) * self.PAGE_SIZE
        return {'photos': [name for _, name in candidates[start:start+self.PAGE_SIZE]],
                'page': page, 'pages': pages, 'total': len(candidates)}

    def detail(self, filename):
        self.open_image(filename)
        metadata = None
        try:
            path = self._inside(Path(filename).with_suffix('.json').name)
            value = json.loads(path.read_text(encoding='utf-8'))
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
            # Only these evidence fields enter the page; never serve arbitrary JSON.
            metadata = {'timestamp': value['timestamp'], 'consecutive_hits': hits, 'detections': rows}
        except (OSError, ValueError, UnicodeError, RecursionError):
            pass
        return {'filename': filename, 'metadata': metadata}
