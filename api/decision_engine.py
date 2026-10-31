"""
Failure Decision Engine & Safety Interlocking Module
===================================================
Applies multi-frame temporal confidence smoothing (consecutive hit counter)
and state interlocking to prevent false alarms before triggering print pause.
"""

import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("DecisionEngine")


class DecisionEngine:
    def __init__(
        self,
        consecutive_required: int = 6,
        confidence_threshold: float = 0.35,
        target_classes: List[str] = None,
        dry_run: bool = True,
        action: str = "pause",
        cooldown_seconds: int = 120
    ):
        self.consecutive_required = consecutive_required
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or ["spaghetti", "warping"]
        self.dry_run = dry_run
        self.action = action.lower()  # "pause" or "stop"
        self.cooldown_seconds = cooldown_seconds

        self.consecutive_hits = 0
        self.last_trigger_time = 0
        self.total_frames_processed = 0
        self.total_failures_detected = 0

    def evaluate_frame(
        self,
        detections: List[Dict[str, Any]],
        gcode_state: str
    ) -> Dict[str, Any]:
        """
        Evaluates detections for a single frame.

        Args:
            detections: List of detection dicts (label, confidence, bbox).
            gcode_state: Current printer state from MQTT (e.g., 'RUNNING', 'PAUSE').

        Returns:
            Dict containing evaluation result:
            {
                "has_failure": bool,
                "consecutive_hits": int,
                "triggered": bool,
                "dry_run": bool,
                "action": str,
                "highest_confidence": float,
                "matching_detections": List[Dict]
            }
        """
        self.total_frames_processed += 1
        now = time.time()

        # State Interlocking: Only evaluate failure if the printer is actually actively printing!
        if gcode_state != "RUNNING":
            if self.consecutive_hits > 0:
                logger.info(f"Printer state is '{gcode_state}' (not RUNNING). Resetting hit counter.")
            self.consecutive_hits = 0
            return {
                "has_failure": False,
                "consecutive_hits": 0,
                "triggered": False,
                "dry_run": self.dry_run,
                "action": self.action,
                "highest_confidence": 0.0,
                "matching_detections": []
            }

        # Filter detections matching target failure classes and confidence threshold
        matching_detections = []
        highest_conf = 0.0

        for det in detections:
            label = det.get("label", "").lower()
            conf = det.get("confidence", 0.0)

            # Check if detection matches target classes
            is_target = any(tc.lower() in label for tc in self.target_classes)
            if is_target and conf >= self.confidence_threshold:
                matching_detections.append(det)
                if conf > highest_conf:
                    highest_conf = conf

        has_failure = len(matching_detections) > 0

        # Update temporal consecutive counter
        if has_failure:
            self.consecutive_hits += 1
            self.total_failures_detected += 1
            logger.warning(
                f"[DETECTION] Failure hit {self.consecutive_hits}/{self.consecutive_required} "
                f"(Max Conf: {highest_conf:.2f}, Matches: {len(matching_detections)})"
            )
        else:
            if self.consecutive_hits > 0:
                logger.info(f"Frame clean. Resetting consecutive hits from {self.consecutive_hits} to 0.")
            self.consecutive_hits = 0

        # Check if confirmed failure threshold reached and cooldown expired
        triggered = False
        in_cooldown = (now - self.last_trigger_time) < self.cooldown_seconds

        if self.consecutive_hits >= self.consecutive_required:
            if not in_cooldown:
                triggered = True
                self.last_trigger_time = now
                logger.critical(
                    f"⚠️ [CONFIRMED FAILURE] Spaghetti failure confirmed across {self.consecutive_hits} consecutive frames! "
                    f"Action: {self.action.upper()} (Dry-Run: {self.dry_run})"
                )
            else:
                logger.info(f"Threshold reached but in cooldown ({int(self.cooldown_seconds - (now - self.last_trigger_time))}s remaining).")

        return {
            "has_failure": has_failure,
            "consecutive_hits": self.consecutive_hits,
            "triggered": triggered,
            "dry_run": self.dry_run,
            "action": self.action,
            "highest_confidence": round(highest_conf, 4),
            "matching_detections": matching_detections
        }
