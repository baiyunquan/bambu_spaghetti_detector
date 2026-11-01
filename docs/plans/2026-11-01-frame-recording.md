# Print frame recording and manual capture

User change to the original read-only gallery: save every received frame while
MQTT reports `RUNNING`, and add a “拍一张” button usable even when idle. Work is
directly on main, without feature branches or worktrees. Existing pause policy,
model, thresholds, dry-run and boot settings remain unchanged.

Use the existing controller camera/UDS loop. A second camera connection or GPU
client adds contention without providing a better current picture; keeping a
cached frame would risk presenting stale data. The chosen implementation
therefore admits one pending capture request and saves the next frame received
after its monotonic request time, with the usual inference results. Requests
time out after 10 seconds and are cancelled on controller shutdown.

`FrameRecorder` saves all RUNNING frames, also on inference errors, and fulfils
manual requests using the same saved frame when printing. Only filenames/events
cross into the web thread; no frame queue/history is retained. Frames are saved
synchronously to keep memory bounded. Failed inference is marked explicitly in
JPEG and JSON and displayed as unavailable, rather than healthy zero detection.
Normal frames carry a neutral header, and confirmed failures retain their
existing separate forensic snapshot. Unique microsecond/random names avoid
overwrites. JSON then JPEG are published using atomic file replacements.

Add `POST /api/photos/capture` and a small local script with pending/error state;
successful capture opens its detail page. A custom request header and same-origin
check prevent cross-site form capture. No additional printer commands are sent.
All files use the current snapshot directory, pagination, and containment rules.
Retention is all images; no automatic deletion or reduced capture frequency.

Validation: recording/count/annotation, idle manual request, no duplicates during
printing, no pre-click frames, cancellation/timeout/busy/shutdown, storage and
inference failures, API origin/errors, controller integration and full suite;
then Nano Python 3.10 tests and a real manual capture through the deployed web UI.
Backup source/config before fast-forward deployment; verify unchanged detection
and safety config, one worker/sampler, real photo metadata and gallery display.
