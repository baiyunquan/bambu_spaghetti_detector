# Embedded web UI validation — 2026-11-01

Implementation runs directly on `main`. The temporary feature worktree and
branch were removed at the user's request. No one-hour replay was performed.

## Automated checks

- Host: 95 passed, 11 hardware-dependent skips, 2 subtests passed; frontend Node
  checks passed. Includes gallery pagination, malformed records, escaping,
  directory containment, monotonic expiry/capacity, subprocess cleanup, port
  conflicts, partial controller startup, and a single shared sampler.
- Nano Python 3.10: 28 gallery, telemetry, lifecycle and worker startup tests
  passed. Existing Python 3.6 inference environment was preserved.

## Nano short replay

`scripts/validate_web.py` uses one UDS client and does not initialize printer
authentication, video or MQTT. Both runs used the existing YOLOv5 416 FP16
engine, confidence 0.30, native CUDA, and the same real 1536×1080 camera JPEG.
Raw measurements and the JPEG hash are in `baseline.json` and `web.json`.
Each run processed 100 frames; P95 excludes the first 10. The web run also
requested the gallery, bundled assets, 24 images, and shared metrics.

| Measurement | Web disabled | Web enabled |
| --- | ---: | ---: |
| Warmed worker startup | 6.51 s | 7.96 s |
| First request | 59.24 ms | 59.72 ms |
| Steady P95 | 59.30 ms | 60.23 ms |
| Controller RSS | 29.59 MiB | 57.78 MiB |
| Owned tegrastats RSS | 0 | 0.86 MiB |
| Worker RSS | 1060.98 MiB | 1062.57 MiB |
| Controller / worker swap | 0 / 0 | 0 / 0 |

Added controller + sampler RSS is **29.05 MiB**, below 100 MiB. P95 increased
**1.58%**, below 10%. Controller modules contained no torch, torchvision or
TensorRT imports. The actual Nano metrics stream was available. System swap
already contained unrelated pages; process swap above is measured separately.

An initial staging run lacked the ignored compiled PTX; the exact production
artifact was copied without rebuilding or changing the GPU code. Another early
run left a non-listening socket after worker exit. This was not reproduced in
the instrumented checks: direct SIGTERM, controller stop after five frames,
and both complete 100-frame runs exited with code 0 and removed their socket
(0.62 / 0.82 seconds for the complete runs). No speculative signal-handling
change or weakened cleanup assertion was introduced. Keep the diagnostic
shutdown output when reproducing any future instance.

## Browser checks

Chrome inspected the Nano staging server through an SSH loopback forward.
Synthetic, visibly marked fixtures stayed in an isolated staging photo folder;
they were not added to production evidence. Verified desktop gallery, 24-item
pagination, full annotated image and detection table, 390×844 mobile layout,
real metrics and Chart.js rendering. Stopping the staging server changed the
page to “连接中断，等待重试” and cleared current readings. Temporary viewport
override and the staging server were stopped afterwards.

## Deployment

Production code, private configuration, model hashes, and the three existing
untracked model metadata files were backed up to
`/home/jetson/spaghetti-web-backup-20261101` before deployment. Metadata hashes
matched the versions already tracked on local main. Deployment verification is
recorded below after the production controller is started.
