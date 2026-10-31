"""
Bambu Lab A1 Local TLS Video Stream Receiver (Port 6000)
========================================================
Connects to the A1 printer's local TLS TCP port 6000, executes the Bambu
handshake protocol, and continuously yields validated 1280x720 JPEG frames.
"""

import ssl
import time
import socket
import struct
import logging
from typing import Generator, Optional

logger = logging.getLogger("VideoStream")


class A1VideoStream:
    FRAME_START = b"\xff\xd8"
    FRAME_END = b"\xff\xd9"

    def __init__(self, printer_ip: str, access_code: str, port: int = 6000, timeout: int = 10):
        self.printer_ip = printer_ip
        self.access_code = access_code
        self.port = port
        self.timeout = timeout
        self.socket: Optional[ssl.SSLSocket] = None
        self._running = False

    def connect(self):
        """Establish TLS TCP connection to port 6000 and authenticate."""
        self.disconnect()
        logger.info(f"Connecting to A1 video stream at {self.printer_ip}:{self.port} (TLS)...")

        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.settimeout(self.timeout)

        # Insecure TLS context (printer uses self-signed BBL CA certificate)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        self.socket = ctx.wrap_socket(raw_sock)
        self.socket.connect((self.printer_ip, self.port))

        # Send authentication packet
        self._send_auth_packet()
        logger.info("Connected and authenticated with A1 video server!")

    def _send_auth_packet(self):
        """Constructs and sends standard 64-byte Bambu authentication packet."""
        username = b"bblp"
        password = self.access_code.encode("ascii")

        # Format: size(4, 0x40) | type(4, 0x3000) | flags(4, 0) | reserved(4, 0) | username(32) | password(32)
        packet = struct.pack("<IIII", 0x40, 0x3000, 0, 0)
        packet += username.ljust(32, b"\x00")
        packet += password.ljust(32, b"\x00")

        self.socket.sendall(packet)

    def _recv_exact(self, num_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionResetError("A1 video stream closed by remote host")
            data.extend(chunk)
        return bytes(data)

    def get_frame(self) -> bytes:
        """Reads one complete validated JPEG frame from the stream."""
        if not self.socket:
            raise ConnectionError("Stream socket not connected")

        # 1. Read 16-byte header
        header = self._recv_exact(16)
        payload_size, itrack, flags, reserved = struct.unpack("<IIII", header)

        # Sanity check frame size (720p JPEG usually 30KB ~ 300KB)
        if payload_size <= 0 or payload_size > 5 * 1024 * 1024:
            raise ValueError(f"Corrupt frame payload size: {payload_size} bytes")

        # 2. Read JPEG image data
        image_data = self._recv_exact(payload_size)

        # 3. Validate JPEG magic bytes
        if not (image_data.startswith(self.FRAME_START) and image_data.endswith(self.FRAME_END)):
            raise ValueError("Invalid JPEG header or trailer magic bytes")

        return image_data

    def _close_socket(self):
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def stream_frames(self) -> Generator[bytes, None, None]:
        """Continuously yields frames, automatically reconnecting on network errors."""
        self._running = True
        backoff = 2

        while self._running:
            try:
                if self.socket is None:
                    self.connect()
                    backoff = 2

                while self._running:
                    frame = self.get_frame()
                    yield frame

            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"Video stream error: {e}. Reconnecting in {backoff}s...")
                self._close_socket()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def disconnect(self):
        self._running = False
        self._close_socket()
