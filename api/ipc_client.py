"""
Unix Domain Socket (UDS) IPC Client
===================================
Communicates with the JetPack Python 3.6 TensorRT inference worker daemon
over /tmp/spaghetti_trt.sock with sub-millisecond latency.
"""

import os
import json
import socket
import struct
import logging
from typing import Dict, Any, List

logger = logging.getLogger("IPCClient")
SOCKET_PATH = "/tmp/spaghetti_trt.sock"


def recv_exact(sock, num_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionResetError("UDS connection closed by inference worker")
        data.extend(chunk)
    return bytes(data)


class InferenceClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 5.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock: socket.socket = None

    def connect(self):
        """Connects to the UDS inference socket."""
        if not os.path.exists(self.socket_path):
            raise FileNotFoundError(f"Inference worker socket not found at {self.socket_path}. Is worker running?")

        self.close()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)

    def detect(self, jpeg_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Sends raw JPEG bytes to the TensorRT worker and returns detection results.

        Returns:
            List of detection dicts: [{"label": "spaghetti", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}, ...]
        """
        if self.sock is None:
            self.connect()

        try:
            # 1. Send 4-byte header + JPEG bytes
            header = struct.pack(">I", len(jpeg_bytes))
            self.sock.sendall(header + jpeg_bytes)

            # 2. Receive 4-byte response header
            resp_header = recv_exact(self.sock, 4)
            resp_len = struct.unpack(">I", resp_header)[0]

            # 3. Receive JSON payload
            resp_bytes = recv_exact(self.sock, resp_len)
            result = json.loads(resp_bytes.decode("utf-8"))

            if result.get("status") != "ok":
                logger.error(f"Inference worker returned error: {result.get('error')}")
                return []

            return result.get("detections", [])

        except Exception as e:
            logger.warning(f"IPC communication error: {e}. Reconnecting...")
            self.close()
            raise

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
