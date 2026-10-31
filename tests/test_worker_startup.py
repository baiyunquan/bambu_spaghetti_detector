"""Controller readiness contract without starting GPU workers or networking."""
import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
HAS_MQTT = importlib.util.find_spec("paho") is not None


@unittest.skipUnless(HAS_MQTT, "Requires the controller environment")
class TestWorkerStartup(unittest.TestCase):
    def setUp(self):
        import main
        self.main = main
        self.app = main.SpaghettiDetectionApp.__new__(main.SpaghettiDetectionApp)
        self.app.config = {"detection": {"model_engine": "models/test.engine"}}
        self.app.running = True
        self.app.worker_process = None
        self.elapsed = 0.0

    def external_dependencies(self, exit_code=None, socket_exists_at=1., listen_at=70.):
        def sleep(seconds):
            self.elapsed += seconds

        owner = self

        class Socket:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def settimeout(self, timeout): pass
            def connect(self, path):
                if owner.elapsed < listen_at:
                    raise ConnectionRefusedError("Not listening yet")

        stack = ExitStack()
        worker = SimpleNamespace(poll=lambda: exit_code, returncode=exit_code, pid=123)
        for target, replacement in [
            ("subprocess.Popen", lambda *a, **k: worker),
            ("time.sleep", sleep),
            ("time.monotonic", lambda: self.elapsed),
            ("os.path.exists", lambda path: self.elapsed >= socket_exists_at),
            ("socket.socket", lambda *a, **k: Socket()),
            ("os.getpgid", lambda pid: pid),
            ("os.killpg", lambda *args: None),
        ]:
            stack.enter_context(patch("main." + target, replacement))
        stack.enter_context(patch.object(Path, "exists", return_value=True))
        return stack

    def test_waits_for_listening_after_slow_full_warmup(self):
        with self.external_dependencies():
            self.app._ensure_worker_running()
        self.assertGreaterEqual(self.elapsed, 70.)
        self.assertLess(self.elapsed, 120.)

    def test_worker_exit_reported_without_waiting_for_deadline(self):
        with self.external_dependencies(exit_code=7, socket_exists_at=999.):
            with self.assertRaises(RuntimeError):
                self.app._ensure_worker_running()
        self.assertLess(self.elapsed, 1.)

    def test_startup_timeout_stops_owned_worker(self):
        with self.external_dependencies(socket_exists_at=999.):
            with self.assertRaises(TimeoutError):
                self.app._ensure_worker_running()
        self.assertFalse(self.app.running)
        self.assertIsNone(self.app.worker_process)


if __name__ == "__main__":
    unittest.main()
