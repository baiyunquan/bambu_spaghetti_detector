"""
GPU Post-Processing & NMS Module for TensorRT Output
====================================================
Performs GPU-accelerated Non-Maximum Suppression (via torchvision.ops.nms)
and maps bounding boxes back to original camera frame coordinates.
"""

from typing import List, Dict, Any
import torch
import torchvision


class GPUPostProcessor:
    def __init__(self, conf_thresh: float = 0.35, iou_thresh: float = 0.45, device: str = "cuda:0"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device)

    def process_obico(self, raw_outputs: List[torch.Tensor], meta: Dict[str, Any], class_names: List[str] = None) -> List[Dict[str, Any]]:
        """
        Post-process Obico model output tensors on GPU.
        raw_outputs:
          - boxes: [1, N, 1, 4] (normalized [x1, y1, x2, y2] in [0, 1])
          - confs: [1, N, 1] (class confidence in [0, 1])
        """
        boxes_tensor, confs_tensor = raw_outputs[0], raw_outputs[1]
        if class_names is None:
            class_names = ["spaghetti"]

        # Squeeze batch and singleton dims: boxes -> [N, 4], confs -> [N]
        boxes = boxes_tensor.squeeze(0).squeeze(1)  # [N, 4]
        confs = confs_tensor.squeeze(0).squeeze(1)  # [N]

        # Filter by confidence threshold on GPU
        mask = confs >= self.conf_thresh
        if not mask.any():
            return []

        filt_boxes = boxes[mask]
        filt_confs = confs[mask]

        # Convert normalized coordinates [0, 1] to letterbox target dimensions [target_w, target_h]
        target_w = meta["target_w"]
        target_h = meta["target_h"]
        x1 = filt_boxes[:, 0] * target_w
        y1 = filt_boxes[:, 1] * target_h
        x2 = filt_boxes[:, 2] * target_w
        y2 = filt_boxes[:, 3] * target_h
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # GPU NMS
        keep = torchvision.ops.nms(xyxy, filt_confs, self.iou_thresh)
        if len(keep) == 0:
            return []

        nms_boxes = xyxy[keep]
        nms_confs = filt_confs[keep]

        # Map back to original image coordinates
        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]

        results = []
        for i in range(len(keep)):
            box = nms_boxes[i]
            bx1 = float(torch.clamp((box[0] - pad_x) / scale, 0, orig_w).item())
            by1 = float(torch.clamp((box[1] - pad_y) / scale, 0, orig_h).item())
            bx2 = float(torch.clamp((box[2] - pad_x) / scale, 0, orig_w).item())
            by2 = float(torch.clamp((box[3] - pad_y) / scale, 0, orig_h).item())
            conf_val = float(nms_confs[i].item())

            results.append({
                "label": class_names[0],
                "class_id": 0,
                "confidence": round(conf_val, 4),
                "bbox": [int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))]
            })

        return results

    def process_yolov8(self, raw_outputs: List[torch.Tensor], meta: Dict[str, Any], class_names: List[str] = None) -> List[Dict[str, Any]]:
        """
        Post-process YOLOv8 model output tensor on GPU.
        output0: [1, 4 + num_classes, num_anchors] (e.g. [1, 6, 8400])
        """
        if class_names is None:
            class_names = ["spaghetti", "warping"]

        output = raw_outputs[0].squeeze(0)  # [4 + num_classes, num_anchors]
        # Transpose to [num_anchors, 4 + num_classes]
        preds = output.permute(1, 0)
        
        boxes = preds[:, :4]  # cx, cy, w, h
        scores = preds[:, 4:]  # class scores

        max_scores, class_ids = torch.max(scores, dim=1)
        mask = max_scores >= self.conf_thresh
        if not mask.any():
            return []

        filt_boxes = boxes[mask]
        filt_scores = max_scores[mask]
        filt_classes = class_ids[mask]

        # Convert cx, cy, w, h to xyxy
        cx = filt_boxes[:, 0]
        cy = filt_boxes[:, 1]
        w = filt_boxes[:, 2]
        h = filt_boxes[:, 3]
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # GPU NMS (batched by class)
        offsets = filt_classes.float() * 4096.0
        keep = torchvision.ops.nms(xyxy + offsets.unsqueeze(1), filt_scores, self.iou_thresh)
        if len(keep) == 0:
            return []

        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]

        results = []
        for idx in keep:
            box = xyxy[idx]
            cid = int(filt_classes[idx].item())
            cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            conf_val = float(filt_scores[idx].item())

            bx1 = float(torch.clamp((box[0] - pad_x) / scale, 0, orig_w).item())
            by1 = float(torch.clamp((box[1] - pad_y) / scale, 0, orig_h).item())
            bx2 = float(torch.clamp((box[2] - pad_x) / scale, 0, orig_w).item())
            by2 = float(torch.clamp((box[3] - pad_y) / scale, 0, orig_h).item())

            results.append({
                "label": cname,
                "class_id": cid,
                "confidence": round(conf_val, 4),
                "bbox": [int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))]
            })

        return results

    def process_yolov5(self, raw_outputs: List[torch.Tensor], meta: Dict[str, Any], class_names: List[str] = None) -> List[Dict[str, Any]]:
        """
        Post-process YOLOv5 model output tensor on GPU.
        output0: [1, num_anchors, 5 + num_classes] (e.g. [1, 10647, 9])
        Columns: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
        """
        if class_names is None:
            class_names = ["error", "extrusor", "part", "spaghetti"]

        preds = raw_outputs[0].squeeze(0)  # [num_anchors, 5 + num_classes]
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)

        boxes = preds[:, :4]  # cx, cy, w, h
        obj_conf = preds[:, 4:5]  # [num_anchors, 1]
        cls_probs = preds[:, 5:]  # [num_anchors, num_classes]
        scores = cls_probs * obj_conf  # [num_anchors, num_classes]

        max_scores, class_ids = torch.max(scores, dim=1)
        mask = max_scores >= self.conf_thresh
        if not mask.any():
            return []

        filt_boxes = boxes[mask]
        filt_scores = max_scores[mask]
        filt_classes = class_ids[mask]

        # Convert cx, cy, w, h to xyxy
        cx = filt_boxes[:, 0]
        cy = filt_boxes[:, 1]
        w = filt_boxes[:, 2]
        h = filt_boxes[:, 3]
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # GPU NMS (batched by class)
        offsets = filt_classes.float() * 4096.0
        keep = torchvision.ops.nms(xyxy + offsets.unsqueeze(1), filt_scores, self.iou_thresh)
        if len(keep) == 0:
            return []

        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]

        results = []
        for idx in keep:
            box = xyxy[idx]
            cid = int(filt_classes[idx].item())
            cname = class_names[cid] if cid < len(class_names) else f"class_{cid}"
            conf_val = float(filt_scores[idx].item())

            bx1 = float(torch.clamp((box[0] - pad_x) / scale, 0, orig_w).item())
            by1 = float(torch.clamp((box[1] - pad_y) / scale, 0, orig_h).item())
            bx2 = float(torch.clamp((box[2] - pad_x) / scale, 0, orig_w).item())
            by2 = float(torch.clamp((box[3] - pad_y) / scale, 0, orig_h).item())

            results.append({
                "label": cname,
                "class_id": cid,
                "confidence": round(conf_val, 4),
                "bbox": [int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))]
            })

        return results

    def process_yolo(self, raw_outputs: List[torch.Tensor], meta: Dict[str, Any], class_names: List[str] = None, model_type: str = None) -> List[Dict[str, Any]]:
        """
        Auto-dispatch to process_yolov5 or process_yolov8 based on output tensor layout.
        YOLOv8: [1, 4 + C, N] where C+4 << N (e.g. 6 << 8400 or 3549)
        YOLOv5: [1, N, 5 + C] where N >> 5 + C (e.g. 10647 >> 9)
        """
        if model_type == "yolov5":
            return self.process_yolov5(raw_outputs, meta, class_names)
        elif model_type == "yolov8":
            return self.process_yolov8(raw_outputs, meta, class_names)

        out = raw_outputs[0]
        if out.ndim == 3:
            if out.shape[1] > out.shape[2] or (out.shape[2] <= 85 and out.shape[1] > 32):
                return self.process_yolov5(raw_outputs, meta, class_names)
        return self.process_yolov8(raw_outputs, meta, class_names)
