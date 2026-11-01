"""Bounded manual requests serviced by the controller's existing frame loop."""
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger('FrameRecorder')


class CaptureUnavailable(RuntimeError):
    pass


class CaptureBusy(RuntimeError):
    pass


class CaptureRequests:
    def __init__(self):
        self._lock = threading.Lock()
        self._available = False
        self._active = None
        self.pending_event = threading.Event()

    def start(self):
        with self._lock:
            self._available = True

    def request(self, timeout=10):
        with self._lock:
            if not self._available:
                raise CaptureUnavailable('相机采集尚未就绪')
            if self._active is not None:
                raise CaptureBusy('已有拍摄请求，请稍后重试')
            ticket = {'event': threading.Event(), 'claimed': False, 'error': None,
                      'requested_at': time.monotonic()}
            self._active = ticket
            self.pending_event.set()
        if not ticket['event'].wait(timeout):
            with self._lock:
                if self._active is ticket:
                    self._active = None
                    self.pending_event.clear()
            raise TimeoutError('未收到新的相机画面，请检查相机连接后重试')
        if ticket['error'] is not None:
            raise ticket['error']
        return ticket['filename']

    def claim(self, frame_received_at):
        with self._lock:
            ticket = self._active
            if (ticket is not None and not ticket['claimed']
                    and frame_received_at >= ticket['requested_at']):
                ticket['claimed'] = True
                self.pending_event.clear()
                return ticket
        return None

    def complete(self, ticket, filename=None, error=None):
        with self._lock:
            if self._active is ticket:
                self._active = None
                self.pending_event.clear()
                ticket.update(filename=filename, error=error)
                ticket['event'].set()

    def stop(self):
        with self._lock:
            self._available = False
            if self._active is not None:
                self._active['error'] = CaptureUnavailable('相机采集已停止')
                self._active['event'].set()
                self._active = None
            self.pending_event.clear()


class FrameRecorder:
    def __init__(self, snapshots, requests):
        self.snapshots, self.requests = snapshots, requests

    def record(self, jpeg_bytes, detections, consecutive_hits, state, printer_info,
               inference_error=None, frame_received_at=None):
        ticket = self.requests.claim(time.monotonic() if frame_received_at is None else frame_received_at)
        if state != 'RUNNING' and ticket is None:
            return None
        try:
            path = Path(self.snapshots.save_frame(
                jpeg_bytes=jpeg_bytes, detections=detections, consecutive_hits=consecutive_hits,
                printer_info=dict(printer_info, gcode_state=state),
                capture_kind='print' if state == 'RUNNING' else 'manual',
                manual_requested=ticket is not None, inference_error=inference_error))
        except Exception as exc:
            logger.exception('Could not save camera frame')
            if ticket is not None:
                self.requests.complete(ticket, error=OSError('照片保存失败，请检查磁盘空间和目录权限'))
            return None
        if ticket is not None:
            self.requests.complete(ticket, filename=path.name)
        return path
