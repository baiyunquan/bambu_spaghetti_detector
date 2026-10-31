"""Independent NumPy decode/NMS oracle for the native CUDA implementation.

Candidate coordinates are in letterboxed input pixels. Columns are
[x1, y1, x2, y2, score, class_id, original_anchor_index]. No top-k cap applies.
"""
import numpy as np

try:
    from .model_contract import canonicalize_labels, validate_model_type
except ImportError:
    from model_contract import canonicalize_labels, validate_model_type


def _threshold(value, name):
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError('{} must be finite and in [0,1]'.format(name))
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('{} must be finite and in [0,1]'.format(name))
    return value


def _metadata(meta):
    keys = ('orig_w', 'orig_h', 'target_w', 'target_h', 'scale', 'pad_x', 'pad_y')
    try:
        result = {key: float(meta[key]) for key in keys}
    except (KeyError, TypeError, ValueError):
        raise ValueError('Incomplete or invalid letterbox metadata')
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Letterbox metadata must be finite')
    if any(result[key] <= 0 for key in keys[:5]):
        raise ValueError('Letterbox dimensions and scale must be positive')
    return result


def _outputs(raw_outputs, model_type):
    validate_model_type(model_type)
    if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != (2 if model_type == 'obico' else 1):
        raise ValueError('Wrong number of raw model outputs')
    outputs = [np.asarray(output) for output in raw_outputs]
    for output in outputs:
        if not np.issubdtype(output.dtype, np.floating) or not np.all(np.isfinite(output)):
            raise ValueError('Model outputs must contain finite floating point values')
    first = outputs[0]
    if model_type == 'obico':
        valid = (first.ndim == 4 and first.shape[0] == 1 and first.shape[2:] == (1, 4) and
                 outputs[1].shape == (1, first.shape[1], 1))
    elif model_type == 'yolov8':
        valid = first.ndim == 3 and first.shape[0] == 1 and first.shape[1] > 4
    else:
        valid = first.ndim == 3 and first.shape[0] == 1 and first.shape[2] > 5
    if not valid:
        raise ValueError('Invalid {} raw output layout'.format(model_type))
    return outputs


def decode_candidates(raw_outputs, model_type, meta, conf_thresh):
    """Decode highest scoring class per anchor without NMS or image clipping."""
    threshold = _threshold(conf_thresh, 'conf_thresh')
    metadata = _metadata(meta)
    outputs = _outputs(raw_outputs, model_type)
    if model_type == 'obico':
        boxes = outputs[0][0, :, 0, :].copy()
        boxes *= np.asarray([metadata['target_w'], metadata['target_h'],
                             metadata['target_w'], metadata['target_h']], dtype=boxes.dtype)
        scores = outputs[1][0, :, 0]
        classes = np.zeros(len(scores), dtype=np.int64)
    else:
        predictions = outputs[0][0].T if model_type == 'yolov8' else outputs[0][0]
        center_boxes = predictions[:, :4]
        class_scores = predictions[:, 4:] if model_type == 'yolov8' else predictions[:, 5:] * predictions[:, 4:5]
        classes = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(predictions)), classes]
        boxes = np.empty_like(center_boxes)
        boxes[:, :2] = center_boxes[:, :2] - center_boxes[:, 2:] * .5
        boxes[:, 2:] = center_boxes[:, :2] + center_boxes[:, 2:] * .5
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(scores)):
        raise ValueError('Decoded candidates must be finite')
    anchors = np.flatnonzero(scores >= threshold)
    if not len(anchors):
        return np.empty((0, 7), dtype=np.float64)
    return np.column_stack((boxes[anchors], scores[anchors], classes[anchors], anchors))


def reference_nms(candidates, iou_thresh):
    """Stable, class-aware greedy NMS. Suppress only when IoU > threshold."""
    threshold = _threshold(iou_thresh, 'iou_thresh')
    candidates = np.asarray(candidates)
    if (candidates.ndim != 2 or candidates.shape[1] != 7 or
            not np.all(np.isfinite(candidates))):
        raise ValueError('Candidates must be a finite Nx7 array')
    order = np.lexsort((candidates[:, 6], -candidates[:, 4]))
    keep = []
    boxes = candidates[:, :4].astype(np.float64)
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    while len(order):
        current = order[0]
        keep.append(current)
        rest = order[1:]
        low = np.maximum(boxes[current, :2], boxes[rest, :2])
        high = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        intersection = np.prod(np.maximum(high - low, 0), axis=1)
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppressed = (candidates[rest, 5] == candidates[current, 5]) & (iou > threshold)
        order = rest[~suppressed]
    return np.asarray(keep, dtype=np.int64)


def decode(raw_outputs, model_type, meta, conf_thresh, iou_thresh, class_names):
    """Return canonical labels and rounded original-frame boxes and scores."""
    names = canonicalize_labels(class_names)
    outputs = _outputs(raw_outputs, model_type)
    class_count = 1 if model_type == 'obico' else outputs[0].shape[1] - 4 if model_type == 'yolov8' else outputs[0].shape[2] - 5
    if len(names) != class_count:
        raise ValueError('class_names do not match model output class count')
    metadata = _metadata(meta)
    candidates = decode_candidates(outputs, model_type, metadata, conf_thresh)
    keep = reference_nms(candidates, iou_thresh)
    results = []
    for index in keep:
        row = candidates[index]
        box = (row[:4] - [metadata['pad_x'], metadata['pad_y'], metadata['pad_x'], metadata['pad_y']]) / metadata['scale']
        box = np.clip(box, 0, [metadata['orig_w'], metadata['orig_h'], metadata['orig_w'], metadata['orig_h']])
        class_id = int(row[5])
        results.append(dict(label=names[class_id], class_id=class_id,
                            confidence=round(float(row[4]), 4),
                            bbox=[int(round(float(value))) for value in box]))
    return results
