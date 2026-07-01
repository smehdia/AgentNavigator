#!/usr/bin/env python3
"""Convert DroidTask completion tasks into DroidTask-Nav v2 tasks.

V2 uses an LLM conversion policy (GPT-4o) adapted from the earlier Android
Control navigation conversion. It rewrites each original DroidTask goal into the
first stable actionable navigation destination, rather than wrapping the
original task-completion wording. This is the converter used for the paper's
DroidTask-Nav results.

The converter keeps the original DroidTask collector data untouched and writes a
parallel collector-style tree (prompts.json + meta.json with navigation prompt
and target metadata) under the output root.

Paths/keys come from the environment (see baselines/.env.example); CLI args win:
  DROIDTASK_COLLECTOR_ROOT  raw DroidTask collector tasks (droidtask_*/run_*/meta.json)
  DROIDTASK_NAV_TASKS_ROOT  converted DroidTask-Nav output tree (read by run_benchmark.py)
  DROIDTASK_ROOT            DroidTask GT dataset root (states/screens for target frames)
  OPENAI_API_KEY            GPT-4o conversion (required unless --dry-run)

Third-party deps: openai (the conversion LLM).

Example:
  python baselines/droidtask_nav/convert.py \
      --source-root "$DROIDTASK_COLLECTOR_ROOT" \
      --out-root "$DROIDTASK_NAV_TASKS_ROOT" \
      --gt-root "$DROIDTASK_ROOT" \
      --model gpt-4o
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# baselines/droidtask_nav/convert.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(os.environ.get(
    "DROIDTASK_COLLECTOR_ROOT", str(REPO_ROOT / "DroidTask_collector_DATA")))
DEFAULT_OUT_ROOT = Path(os.environ.get(
    "DROIDTASK_NAV_TASKS_ROOT", str(REPO_ROOT / "DroidTask_nav_v2_DATA")))
DEFAULT_GT_ROOT = Path(os.environ.get(
    "DROIDTASK_ROOT", str(REPO_ROOT / "DroidTask_GT" / "DroidTask")))


DROIDTASK_NAV_V2_PROMPT = """You are converting DroidTask mobile task-completion goals into navigation-only benchmark goals.

## Original Task
App: {app_label}
Goal: "{goal}"

## Trajectory
Each numbered item is the visible screen BEFORE an action in the ground-truth task trajectory.
The selected element and typed input, if any, describe the action taken from that screen.

{steps}

## DroidTask-Nav v2 Policy

Convert the original task into the FIRST STABLE ACTIONABLE NAVIGATION DESTINATION needed to begin completing the task.
Do NOT simply prepend "Navigate to..." to the full original task.

The navigation goal must describe where the user should arrive, not the final side effect.
Use concise wording like:
- "Navigate to the page where the user can add South Africa holidays to the calendar."
- "Navigate to the screen where the user can choose the alarm snooze duration."
- "Navigate to the search field in the Gallery app."
- "Navigate to the confirmation screen for deleting a contact."

## Key Rules

1. First objective only:
   If the original task chains multiple objectives, convert only the first actionable objective unless a later clause is required to reach it.
   Example: "Add the holidays of South Africa to the calendar and list all events in 2023"
   becomes "Navigate to the page where the user can add South Africa holidays to the calendar."
   Ignore the later "list events" objective for the navigation goal.

2. Stop before task completion:
   Stop at the stable page, dialog, picker, confirmation screen, or control where the requested action can be performed.
   Do not include the final click that changes, saves, sends, deletes, calls, exports, records, or confirms.

3. Text and search:
   For user-specific text entry or search queries, stop at the generic input/search screen before text is entered.
   Search results after a typed query are dynamic unless the result/category is stable app taxonomy.

4. Stable app content can be a valid target:
   Built-in app taxonomy, settings rows, pickers, dialogs, app-provided lists, and fixed options are stable navigation targets.
   Examples: country holiday list, audio source picker, extension picker, sort order picker, theme color page.

5. User/device-specific content is dynamic:
   User-created files, contacts, messages, notes, photos, recordings, arbitrary search results, account-specific data, and typed values are dynamic.
   Stop before selecting or editing that content unless the task is simply to navigate to a generic list/search entry point.

6. Use the trajectory evidence:
   Choose target_step_index as the numbered trajectory item whose PRE-ACTION screen is the navigation target.
   If the final terminal item is already the navigation destination, you may choose that terminal item.
   If no clean navigation target appears in the trajectory, set needs_manual_review=true and explain why.

7. Make the rewritten navigation goal specific:
   Preserve the stable app intent ("South Africa holidays", "snooze duration", "audio source").
   Remove downstream completion wording ("save it", "send it", "list all events", "call him") unless that downstream screen is the first navigation destination.

## Output JSON
Respond with ONLY a JSON object:
{{
  "navigation_goal": "<concise navigation-only goal>",
  "primary_objective": "<the first actionable objective extracted from the original task>",
  "target_step_index": <integer index into the trajectory items>,
  "navigation_steps": [<integer indices that are navigation steps, usually 0..target_step_index>],
  "task_steps": [<integer indices that are task-specific or downstream completion steps>],
  "target_state_hint": "<short visible text/control hint that should appear on the target screen>",
  "target_screen_description": "<brief description of the target screen>",
  "conversion_type": "<one of: already_navigation, input_entry, stable_picker_or_dialog, stable_settings_or_control, stable_content_list, dynamic_content_entry, confirmation_screen, needs_review>",
  "confidence": <0.0-1.0>,
  "needs_manual_review": <true or false>,
  "review_reason": "<brief reason, empty string if no review needed>"
}}"""


@dataclass
class ConversionV2:
    app_dir: str
    task_id: str
    package: str
    original_goal: str
    navigation_goal: str
    primary_objective: str
    original_step_instructions: list[str]
    navigation_step_instructions: list[str]
    original_step_count: int
    navigation_step_count: int
    target_screen_description: str
    target_state_hint: str
    cutoff_step_index: int
    target_state_hash: str | None
    target_screenshot: str | None
    conversion_type: str
    confidence: float
    needs_manual_review: bool
    review_reason: str
    source_task_dir: str
    source_meta_path: str
    llm_raw: dict[str, Any]


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)


def normalize_input(value: Any) -> str:
    if value is None:
        return "null"
    return str(value).strip()


def is_terminal_step(step: dict[str, Any]) -> bool:
    return step.get("Choice") == -1


def is_text_input_step(step: dict[str, Any]) -> bool:
    return normalize_input(step.get("Input")).lower() not in {"", "none", "null"}


def first_state_hash(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            if item:
                return str(item)
        return None
    return str(value) if value else None


def state_hashes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []


def app_label_from_dir(app_dir: str) -> str:
    name = app_dir.removeprefix("droidtask_")
    name = re.sub(r"[_-]\d.*$", "", name)
    return name.replace("_", " ").title()


def app_name_from_source_task(meta: dict[str, Any], app_dir: str) -> str:
    source = meta.get("source_task_file")
    if source:
        parts = Path(source).parts
        if "DroidTask" in parts:
            idx = parts.index("DroidTask")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return app_dir.removeprefix("droidtask_").split("_")[0]


def load_state_index(gt_root: Path, app_name: str) -> dict[str, dict[str, str]]:
    state_dir = gt_root / app_name / "states"
    index: dict[str, dict[str, str]] = {}
    if not state_dir.is_dir():
        return index

    for path in state_dir.glob("state_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        state_hash = data.get("state_str")
        tag = data.get("tag")
        if not state_hash or not tag:
            continue
        screen = state_dir / f"screen_{tag}.png"
        if screen.is_file():
            index[str(state_hash)] = {
                "state_path": str(path),
                "screen_path": str(screen),
                "text": json.dumps(data, ensure_ascii=False),
            }
    return index


def compact_state(state: str, max_chars: int = 1800) -> str:
    state = re.sub(r"\s+", " ", str(state or "")).strip()
    state = re.sub(r"\s*id=\d+\s*", " ", state)
    return state[:max_chars]


def action_instruction(step: dict[str, Any]) -> str:
    if is_terminal_step(step):
        return "Finish task"
    choice = step.get("Choice")
    if is_text_input_step(step):
        return f"Enter text into element {choice}: {normalize_input(step.get('Input'))}"
    return f"Select element {choice}"


def selected_element_text(step: dict[str, Any]) -> str:
    choice = step.get("Choice")
    if choice is None or choice == -1:
        return ""
    state = str(step.get("State") or "")
    pattern = re.compile(
        rf"<(?P<tag>\w+)[^>]*\bid={re.escape(str(choice))}\b[^>]*>(?P<body>.*?)</(?P=tag)>",
        re.DOTALL,
    )
    match = pattern.search(state)
    if not match:
        return ""
    text = re.sub(r"<br\s*/?>", " ", match.group("body"), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tag_start = match.group(0).split(">", 1)[0] + ">"
    return f"{tag_start}{text}</{match.group('tag')}>"


def original_step_instructions(traj: list[dict[str, Any]]) -> list[str]:
    return [action_instruction(step) for step in traj if not is_terminal_step(step)]


def navigation_step_instructions(traj: list[dict[str, Any]], nav_indices: list[int]) -> list[str]:
    out = []
    for idx in nav_indices:
        if 0 <= idx < len(traj) and not is_terminal_step(traj[idx]):
            out.append(action_instruction(traj[idx]))
    return out


def format_steps_for_prompt(traj: list[dict[str, Any]]) -> str:
    chunks = []
    for idx, step in enumerate(traj):
        action = action_instruction(step)
        selected = selected_element_text(step)
        state = compact_state(step.get("State") or "")
        chunk = (
            f"Step {idx}\n"
            f"Pre-action visible state:\n{state}\n"
            f"Action from this screen: {action}\n"
        )
        if selected:
            chunk += f"Selected element content: {selected}\n"
        chunks.append(chunk)
    return "\n".join(chunks)


def call_openai(prompt: str, model: str) -> str:
    # Key comes from the environment only (OPENAI_API_KEY); no key-file fallback.
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy baselines/.env.example to baselines/.env "
            "and set OPENAI_API_KEY (see baselines/README.md)."
        )
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You convert mobile task-completion trajectories into navigation-only benchmark targets."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = [p.strip() for p in text.split("```") if p.strip()]
        for part in parts:
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            return json.loads(text[start:end])
        raise


def call_llm_with_retries(prompt: str, model: str, retries: int = 4) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return extract_json(call_openai(prompt, model))
        except Exception as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM conversion failed after {retries} attempts: {last_exc}")


def normalize_indices(value: Any, max_len: int) -> list[int]:
    out = []
    if isinstance(value, list):
        for item in value:
            try:
                idx = int(item)
            except Exception:
                continue
            if 0 <= idx < max_len and idx not in out:
                out.append(idx)
    return out


def choose_state_for_step(
    step: dict[str, Any],
    state_index: dict[str, dict[str, str]],
    hint: str,
) -> tuple[str | None, str | None]:
    hashes = state_hashes(step.get("state_str"))
    if not hashes:
        return None, None
    if len(hashes) == 1 or not hint.strip():
        selected = hashes[0]
        entry = state_index.get(selected)
        return selected, entry["screen_path"] if entry else None

    hint_tokens = {
        tok.lower()
        for tok in re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", hint)
        if tok.lower() not in {"the", "and", "where", "user", "screen", "page"}
    }
    best_hash = hashes[0]
    best_score = -1
    for candidate in hashes:
        entry = state_index.get(candidate)
        text = (entry or {}).get("text", "").lower()
        score = sum(1 for tok in hint_tokens if tok in text)
        if score > best_score:
            best_score = score
            best_hash = candidate
    entry = state_index.get(best_hash)
    return best_hash, entry["screen_path"] if entry else None


def fallback_conversion(meta_path: Path, source_root: Path, gt_root: Path, error: str) -> ConversionV2:
    # Self-contained fallback when the LLM v2 conversion fails: keep the original
    # goal as the (unconverted) navigation goal and flag it for manual review.
    meta = json.loads(meta_path.read_text())
    task_dir = meta_path.parent
    goal = str(meta.get("prompt") or (meta.get("prompts") or [""])[0]).strip()
    traj = meta.get("gt_trajectory") or []
    return ConversionV2(
        app_dir=task_dir.parent.name,
        task_id=task_dir.name,
        package=str(meta.get("package") or ""),
        original_goal=goal,
        navigation_goal=goal,
        primary_objective=goal,
        original_step_instructions=original_step_instructions(traj),
        navigation_step_instructions=[],
        original_step_count=sum(1 for step in traj if not is_terminal_step(step)),
        navigation_step_count=0,
        target_screen_description="",
        target_state_hint="",
        cutoff_step_index=0,
        target_state_hash=None,
        target_screenshot=None,
        conversion_type="needs_review",
        confidence=0.0,
        needs_manual_review=True,
        review_reason=f"LLM v2 conversion failed: {error}",
        source_task_dir=_safe_relpath(task_dir, source_root),
        source_meta_path=_safe_relpath(meta_path, REPO_ROOT),
        llm_raw={"error": error},
    )


def convert_task_v2(meta_path: Path, source_root: Path, gt_root: Path, model: str) -> ConversionV2:
    meta = json.loads(meta_path.read_text())
    task_dir = meta_path.parent
    app_dir = task_dir.parent.name
    app_label = app_label_from_dir(app_dir)
    app_name = app_name_from_source_task(meta, app_dir)
    state_index = load_state_index(gt_root, app_name)

    goal = str(meta.get("prompt") or (meta.get("prompts") or [""])[0]).strip()
    traj = meta.get("gt_trajectory") or []
    prompt = DROIDTASK_NAV_V2_PROMPT.format(
        app_label=app_label,
        goal=goal,
        steps=format_steps_for_prompt(traj),
    )
    raw = call_llm_with_retries(prompt, model=model)

    try:
        cutoff = int(raw.get("target_step_index", 0))
    except Exception:
        cutoff = 0
    if traj:
        cutoff = max(0, min(cutoff, len(traj) - 1))
    else:
        cutoff = 0

    nav_indices = normalize_indices(raw.get("navigation_steps"), len(traj))
    if not nav_indices and traj:
        nav_indices = list(range(cutoff + 1))

    hint = str(raw.get("target_state_hint") or raw.get("target_screen_description") or "")
    target_hash = None
    target_screen = None
    target_state = ""
    if traj:
        target_hash, target_screen = choose_state_for_step(traj[cutoff], state_index, hint)
        target_state = str(traj[cutoff].get("State") or "")

    needs_review = bool(raw.get("needs_manual_review", False))
    review_reason = str(raw.get("review_reason") or "")
    confidence = float(raw.get("confidence", 0.0) or 0.0)
    if not target_screen:
        needs_review = True
        confidence = min(confidence, 0.5)
        review_reason = (review_reason + " " if review_reason else "") + "No target screenshot was resolved."

    return ConversionV2(
        app_dir=app_dir,
        task_id=task_dir.name,
        package=str(meta.get("package") or ""),
        original_goal=goal,
        navigation_goal=str(raw.get("navigation_goal") or goal).strip(),
        primary_objective=str(raw.get("primary_objective") or "").strip(),
        original_step_instructions=original_step_instructions(traj),
        navigation_step_instructions=navigation_step_instructions(traj, nav_indices),
        original_step_count=sum(1 for step in traj if not is_terminal_step(step)),
        navigation_step_count=len([idx for idx in nav_indices if 0 <= idx < len(traj) and not is_terminal_step(traj[idx])]),
        target_screen_description=str(raw.get("target_screen_description") or compact_state(target_state)).strip(),
        target_state_hint=hint.strip(),
        cutoff_step_index=cutoff,
        target_state_hash=target_hash,
        target_screenshot=target_screen,
        conversion_type=str(raw.get("conversion_type") or "needs_review"),
        confidence=round(max(0.0, min(confidence, 1.0)), 2),
        needs_manual_review=needs_review,
        review_reason=review_reason.strip(),
        source_task_dir=_safe_relpath(task_dir, source_root),
        source_meta_path=_safe_relpath(meta_path, REPO_ROOT),
        llm_raw=raw,
    )


def write_task(out_root: Path, conversion: ConversionV2, source_meta: dict[str, Any]) -> None:
    out_dir = out_root / conversion.app_dir / conversion.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.json").write_text(
        json.dumps({"prompts": [conversion.navigation_goal]}, indent=2) + "\n"
    )
    nav_meta = dict(source_meta)
    nav_meta.update(
        {
            "source": "droidtask-nav-v2",
            "source_droidtask_prompt": conversion.original_goal,
            "prompt": conversion.navigation_goal,
            "prompts": [conversion.navigation_goal],
            "droidtask_nav": asdict(conversion),
            "gt_nav_state_str": conversion.target_state_hash,
            "gt_nav_screenshot": conversion.target_screenshot,
            "gt_nav_target_screen_description": conversion.target_screen_description,
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(nav_meta, indent=2) + "\n")


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            flat[key] = json.dumps(value, ensure_ascii=True)
        else:
            flat[key] = value
    return flat


def write_summaries(out_root: Path, conversions: list[ConversionV2]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in conversions]
    with (out_root / "droidtask_nav_conversions.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    if rows:
        with (out_root / "droidtask_nav_conversions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(flatten_for_csv(row))

    by_type: dict[str, int] = {}
    by_app: dict[str, int] = {}
    review = 0
    for item in conversions:
        by_type[item.conversion_type] = by_type.get(item.conversion_type, 0) + 1
        by_app[item.app_dir] = by_app.get(item.app_dir, 0) + 1
        review += int(item.needs_manual_review)

    lines = [
        "# DroidTask-Nav v2 Conversion Summary",
        "",
        f"- Total converted tasks: {len(conversions)}",
        f"- Needs manual review: {review}",
        f"- Ready without review flag: {len(conversions) - review}",
        "",
        "## By Conversion Type",
        "",
    ]
    for key, count in sorted(by_type.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## By App", ""])
    for key, count in sorted(by_app.items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## v2 Policy",
            "",
            "The navigation goal targets the first stable actionable destination needed",
            "to begin the original DroidTask. Downstream state-changing or dynamic",
            "completion clauses are intentionally removed from the navigation goal.",
        ]
    )
    (out_root / "DROIDTASK_NAV_SUMMARY.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                        help="Raw DroidTask collector tasks (env: DROIDTASK_COLLECTOR_ROOT).")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="Converted DroidTask-Nav output tree (env: DROIDTASK_NAV_TASKS_ROOT).")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT,
                        help="DroidTask GT dataset root (env: DROIDTASK_ROOT).")
    parser.add_argument("--app", help="Only convert one collector app directory.")
    parser.add_argument("--task", help="Only convert one task directory name.")
    parser.add_argument("--limit", type=int, help="Convert only the first N matching tasks.")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fallback-on-error", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    gt_root = args.gt_root.resolve()
    meta_paths = sorted(source_root.glob("droidtask_*/run_*/meta.json"))
    if args.app:
        meta_paths = [p for p in meta_paths if p.parent.parent.name == args.app]
    if args.task:
        meta_paths = [p for p in meta_paths if p.parent.name == args.task]
    if args.limit:
        meta_paths = meta_paths[: args.limit]

    conversions: list[ConversionV2] = []
    for i, meta_path in enumerate(meta_paths, 1):
        print(f"[{i}/{len(meta_paths)}] {meta_path.parent.parent.name}/{meta_path.parent.name}", flush=True)
        try:
            conversion = convert_task_v2(meta_path, source_root, gt_root, model=args.model)
        except Exception as exc:
            if not args.fallback_on_error:
                raise
            conversion = fallback_conversion(meta_path, source_root, gt_root, str(exc))
        conversions.append(conversion)
        print(
            f"  cutoff={conversion.cutoff_step_index} conf={conversion.confidence:.2f} "
            f"review={conversion.needs_manual_review} -> {conversion.navigation_goal}",
            flush=True,
        )
        if not args.dry_run:
            source_meta = json.loads(meta_path.read_text())
            write_task(out_root, conversion, source_meta)

    if not args.dry_run:
        write_summaries(out_root, conversions)
    review = sum(1 for item in conversions if item.needs_manual_review)
    print(f"converted={len(conversions)} needs_manual_review={review} out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
