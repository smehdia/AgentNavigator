# baselines/_common

Shared adb + task plumbing used by the baseline benchmark adapters. `nav_core.py`
provides the `APP_REGISTRY` (app → package/activity/task-folder), a thin `AdbDriver`
(screenshot, tap/swipe/text, app launch, UI-hierarchy dump), `execute_action`
(maps a UI-TARS-style action string to a device action, scaling model coordinates
back to native resolution), and `discover_prompt` (reads a task's goal from
`chosen_target.json`/`prompts.json`). It is a trimmed, sanitized extraction from the
TAG-Nav navigation runner — stdlib only, except `opencv-python` (`cv2`) which is
imported lazily inside `AdbDriver.screenshot_bgr()`.
