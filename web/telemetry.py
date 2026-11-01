"""Single owned tegrastats reader with a bounded monotonic-time history."""
from collections import deque
import logging
import re
import subprocess
import threading
import time

logger = logging.getLogger('JetsonMetrics')
FIELDS = ('cpu_percent', 'gpu_percent', 'ram_used_mb', 'ram_total_mb',
          'swap_used_mb', 'swap_total_mb')


def parse_tegrastats(line):
    result = dict.fromkeys(FIELDS)
    result['temperatures'] = {}
    useful = False
    for field, prefix in (('ram', 'RAM'), ('swap', 'SWAP')):
        match = re.search(r'\b' + prefix + r'\s+(\d+)/(\d+)MB\b', line)
        if match:
            result[field+'_used_mb'], result[field+'_total_mb'] = map(int, match.groups())
            useful = True
    match = re.search(r'\bCPU\s+\[([^]]*)\]', line)
    if match:
        online = []
        for core in match.group(1).split(','):
            reading = re.fullmatch(r'\s*(\d+(?:\.\d+)?)%(?:@\d+)?\s*', core)
            if reading and 0 <= float(reading.group(1)) <= 100:
                online.append(float(reading.group(1)))
        result['cpu_percent'] = sum(online)/len(online) if online else None
        useful = True
    match = re.search(r'\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%', line)
    if match and 0 <= float(match.group(1)) <= 100:
        result['gpu_percent'] = float(match.group(1))
        useful = True
    for name, value in re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)@(-?\d+(?:\.\d+)?)C\b', line):
        result['temperatures'][name] = float(value)
        useful = True
    return result if useful else None


class TelemetrySampler:
    def __init__(self, command=None, clock=time.monotonic, popen=subprocess.Popen, retry_seconds=5):
        self.command = command if command is not None else ['/usr/bin/tegrastats', '--interval', '2000']
        self.clock, self.popen, self.retry_seconds = clock, popen, retry_seconds
        self._history = deque(maxlen=450)
        self._last_success = None
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._stop = threading.Event()
        self._thread = self._process = None
        self._stopped = False
        self._error = None

    @property
    def pid(self):
        with self._lock:
            return self._process.pid if self._process else None

    def _prune(self, now):
        while self._history and now-self._history[0]['_time'] > 900:
            self._history.popleft()

    def ingest(self, line):
        value = parse_tegrastats(line)
        if value is None:
            return False
        now = self.clock()
        with self._lock:
            self._prune(now)
            self._history.append(dict(value, _time=now))
            self._last_success, self._error = now, None
        return True

    def snapshot(self):
        now = self.clock()
        with self._lock:
            self._prune(now)
            samples = [{**{k:v for k,v in row.items() if k != '_time'},
                        'temperatures':dict(row['temperatures']), 'age_s':max(0., now-row['_time'])}
                       for row in self._history]
            fresh = (not self._stopped and bool(samples) and self._last_success is not None
                     and now-self._last_success <= 6)
            status = 'ok' if fresh else ('stale' if samples and not self._stopped else 'unavailable')
            return {'status':status, 'current':samples[-1] if fresh else None,
                    'samples':samples, 'message':'采集正常' if fresh else '采集不可用，等待更新'}

    def start(self):
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._stopped = False
            self._thread = threading.Thread(target=self._collect, name='tegrastats-reader', daemon=True)
            self._thread.start()

    @staticmethod
    def _reap(process):
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
        except ProcessLookupError:
            pass

    def _failed(self, error):
        with self._lock:
            changed = self._error != error
            self._error = error
        if changed:
            logger.warning('Jetson telemetry unavailable: %s', error)

    def _collect(self):
        while not self._stop.is_set():
            process = None
            try:
                process = self.popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                                     bufsize=1, start_new_session=True)
                with self._lock:
                    self._process = process
                if not self._stop.is_set():
                    for line in process.stdout:
                        if self._stop.is_set():
                            break
                        self.ingest(line)
                if not self._stop.is_set():
                    self._failed('tegrastats exited')
            except Exception as exc:
                self._failed(str(exc))
            finally:
                self._reap(process)
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                with self._lock:
                    if self._process is process:
                        self._process = None
            if self._stop.wait(self.retry_seconds):
                break

    def stop(self):
        with self._lifecycle:
            self._stop.set()
            with self._lock:
                self._stopped = True
                process = self._process
            self._reap(process)
            if self._thread is not None:
                self._thread.join(timeout=3)
                if self._thread.is_alive():
                    logger.error('Jetson telemetry reader did not stop')
