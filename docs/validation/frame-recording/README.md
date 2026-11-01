# Print recording / manual capture validation — 2026-11-01

Implemented on main (`dfcad00`) without branches or worktrees. Code review
identified a storage-full path that could abort before confirmed pause dispatch;
a controller regression reproduced it, and the fix preserves the action even if
forensic saving fails. Tests use fake MQTT and never send printer commands.

- Host suite: 113 passed, 11 hardware-dependent skips, 2 subtests passed.
- Both Node chart and capture-button tests passed. Capture tests verify pending
  state, POST header, detail navigation, failure message and retry.
- Nano Python 3.10: 48 recording, capture API, gallery, telemetry, controller
  lifecycle, worker startup and invalid-inference tests passed.
- Covers every RUNNING frame, same-second uniqueness, annotated pixels, idle
  capture, no duplicate printing/manual frame, no pre-click frame, one pending
  request, cancellation/timeout/shutdown, disk-full errors, failed/empty-message
  inference markers and cross-origin rejection.
- Nano offline recording of 20 real 1536×1080 JPEG frames saved exactly 20 unique
  JPEG/JSON pairs: mean 60.68 ms, max 88.83 ms per save. This used an isolated
  temporary directory and simulated RUNNING state; it did not alter printer
  status or create artificial production print history.

Before deployment, tracked source, private configuration, original HEAD and
active engine/PTX hashes were backed up under
`/home/jetson/spaghetti-capture-backup-20261101`. Service was stopped for a clean
fast-forward and restarted after Nano tests. Configuration matched the backup
exactly, including YOLOv5 416/native CUDA, confidence 0.30,
`spaghetti/warping/error`, and dry-run. Engine and compiled PTX hashes matched.
Autostart remains disabled. Controller owns exactly one worker and tegrastats;
none load torch libraries.

Real production capture POST returned **201 in 0.405 seconds**, producing a
1536×1080 annotated JPEG and valid JSON while the printer reported FINISH.
There were zero detections, correctly leaving the image without boxes. The
saved image bytes, photo list and manual detail page were verified over HTTP;
the photo remains in production for the user. Controller RSS after capture was
62.08 MiB. Sanitized deployment details are in `deployment.json`.

The browser provider was unavailable during this follow-up, so the deployed
button was validated by its Node interaction tests plus the real HTTP capture
and page/media checks, rather than a fresh rendered browser interaction. Earlier
desktop/mobile gallery layout checks remain documented in `../web-ui/`.
The device clock is still behind the client clock, so filename timestamps follow
the Nano clock; no system clock changes were made as part of capture deployment.

No real printing job was started for testing. Automatic printing capture was
verified in controller integration tests and the isolated Nano recording replay.
No one-hour tests were run. Rollback can restore the tracked code archive and
private config from the backup after stopping the service; retain all photos.
