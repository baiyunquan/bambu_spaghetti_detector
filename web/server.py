"""One ASGI event loop embedded in the existing Python controller process."""
import logging
import threading
import time

import uvicorn

from .app import create_app
from .telemetry import TelemetrySampler

logger = logging.getLogger('Web')


class EmbeddedWebServer:
    def __init__(self, snapshot_dir, host='0.0.0.0', port=8000, sampler=None, capture=None):
        self.host, self.port = host, port
        self.sampler = sampler if sampler is not None else TelemetrySampler()
        self._server = uvicorn.Server(uvicorn.Config(
            create_app(snapshot_dir, sampler=self.sampler, capture=capture), host=host, port=port,
            loop='asyncio', http='h11', ws='none', workers=1, reload=False,
            access_log=False, log_level='warning', use_colors=False,
            timeout_graceful_shutdown=3, timeout_keep_alive=2, limit_concurrency=32))
        self._thread = None
        self._lock = threading.RLock()
        self._error = None
        self._done = threading.Event()

    @property
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        try:
            # Uvicorn does not install signal handlers outside the main thread.
            self._server.run()
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                self.sampler.stop()
            finally:
                self._done.set()

    def start(self):
        with self._lock:
            if self.is_alive:
                return
            self._done.clear()
            self._error = None
            self._server.should_exit = self._server.force_exit = self._server.started = False
            self._thread = threading.Thread(target=self._run, name='web-server', daemon=True)
            self._thread.start()
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            if self._done.is_set():
                raise RuntimeError('Web server failed to start: {}'.format(self._error or 'server exited'))
            if self._server.started:
                self.port = self._server.servers[0].sockets[0].getsockname()[1]
                logger.info('Read-only web listening on %s:%s', self.host, self.port)
                return
            time.sleep(.01)
        self.stop()
        raise TimeoutError('Web server startup exceeded 5 seconds')

    def stop(self):
        with self._lock:
            self._server.should_exit = True
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=6)
            if thread.is_alive():
                self._server.force_exit = True
                thread.join(timeout=1)
            if thread.is_alive():
                logger.error('Web event loop did not stop within deadline')
        self.sampler.stop()
