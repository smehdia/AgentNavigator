"""Render trajectory MP4 frames to match the inference GUI chat layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

BASE_CANVAS_WIDTH = 900
DEFAULT_CANVAS_WIDTH = 1920

# Tailwind palette used in gui_demo/web
COLOR_PAGE_BG_BGR = (249, 245, 241)  # slate-100 #f1f5f9
COLOR_CARD_BG_RGB = (248, 250, 252)  # slate-50
COLOR_BORDER_RGB = (226, 232, 240)  # slate-200
COLOR_ACCENT_RGB = (99, 102, 241)  # indigo-500
COLOR_TEXT_RGB = (30, 41, 59)  # slate-800
COLOR_MUTED_RGB = (100, 116, 139)  # slate-500
COLOR_USER_BG_RGB = (99, 102, 241)
COLOR_WHITE_RGB = (255, 255, 255)


@dataclass(frozen=True)
class RenderLayout:
    """Scaled layout derived from the 900px web chat design."""

    canvas_width: int

    @property
    def scale(self) -> float:
        return self.canvas_width / BASE_CANVAS_WIDTH

    def dim(self, px: float) -> int:
        return max(1, int(round(px * self.scale)))

    @property
    def card_margin_x(self) -> int:
        return self.dim(24)

    @property
    def card_padding(self) -> int:
        return self.dim(16)

    @property
    def card_radius(self) -> int:
        return self.dim(16)

    @property
    def inner_width(self) -> int:
        return self.canvas_width - 2 * self.card_margin_x - 2 * self.card_padding

    @property
    def step_shot_max_h(self) -> int:
        # Web preview uses ~256px; export uses a much taller viewport for device screenshots.
        return self.dim(720)

    @property
    def candidate_thumb_h(self) -> int:
        return self.dim(320)

    @property
    def candidate_tile_text_h(self) -> int:
        return self.dim(72)


def get_render_layout(canvas_width: Optional[int] = None) -> RenderLayout:
    if canvas_width is None:
        raw = os.environ.get("AGENTNAV_TRAJECTORY_MP4_WIDTH", str(DEFAULT_CANVAS_WIDTH)).strip()
        try:
            canvas_width = int(raw)
        except ValueError:
            canvas_width = DEFAULT_CANVAS_WIDTH
    canvas_width = max(BASE_CANVAS_WIDTH, min(canvas_width, 3840))
    return RenderLayout(canvas_width=canvas_width)


# Backwards-compatible alias for tests/imports.
CANVAS_WIDTH = DEFAULT_CANVAS_WIDTH


def _load_font(size: int, *, bold: bool = False, mono: bool = False):
    from PIL import ImageFont

    env_font = os.environ.get("AGENTNAV_OVERLAY_FONT", "").strip()
    if env_font and os.path.isfile(env_font):
        return ImageFont.truetype(env_font, size)

    if mono:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        ]
    elif bold:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/PingFang.ttc",
        ]

    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []

    lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current: list[str] = []
        for word in words:
            trial = " ".join(current + [word])
            if draw.textlength(trial, font=font) <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
    return lines


def _line_height(font, draw, sample: str = "Ag") -> int:
    box = draw.textbbox((0, 0), sample, font=font)
    return max(1, box[3] - box[1])


def _fit_image_contain_bgr(img_bgr: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, int, int]:
    h, w = img_bgr.shape[:2]
    if w <= 0 or h <= 0:
        placeholder = np.full((max_h, max_w, 3), 241, dtype=np.uint8)
        return placeholder, max_w, max_h

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((max_h, max_w, 3), 241, dtype=np.uint8)
    x0 = (max_w - new_w) // 2
    y0 = (max_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas, max_w, max_h


def _load_candidate_screenshot(logs_root: str, node_id: str) -> Optional[np.ndarray]:
    if not logs_root or not node_id:
        return None
    path = os.path.join(logs_root, "screenshots", f"{node_id}.jpg")
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path)
    return img if img is not None else None


def _bgr_to_pil(img_bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _pil_to_bgr(img_rgb):
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)


def _draw_card(
    draw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    layout: RenderLayout,
    *,
    selected: bool = False,
) -> None:
    outline = COLOR_ACCENT_RGB if selected else COLOR_BORDER_RGB
    width = layout.dim(3) if selected else 1
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=layout.card_radius,
        fill=COLOR_CARD_BG_RGB,
        outline=outline,
        width=width,
    )


def render_query_frame(query: str, layout: Optional[RenderLayout] = None) -> np.ndarray:
    from PIL import Image, ImageDraw

    layout = layout or get_render_layout()
    canvas_width = layout.canvas_width
    font = _load_font(layout.dim(14))
    dummy = Image.new("RGB", (canvas_width, 10))
    draw = ImageDraw.Draw(dummy)
    inner_w = layout.inner_width
    lines = _wrap_text(query, font, int(inner_w * 0.85), draw)
    line_h = _line_height(font, draw)
    pad = layout.card_padding

    bubble_w = min(inner_w, max((draw.textlength(line, font=font) for line in lines), default=80) + 2 * pad)
    bubble_h = max(line_h, len(lines) * (line_h + layout.dim(4)) + 2 * pad)
    total_h = bubble_h + layout.dim(48)

    canvas = Image.new("RGB", (canvas_width, total_h), (241, 245, 249))
    draw = ImageDraw.Draw(canvas)

    x1 = canvas_width - layout.card_margin_x
    x0 = x1 - bubble_w
    y0 = layout.dim(20)
    y1 = y0 + bubble_h
    draw.rounded_rectangle([x0, y0, x1, y1], radius=layout.card_radius, fill=COLOR_USER_BG_RGB)

    y = y0 + pad
    for line in lines:
        draw.text((x0 + pad, y), line, font=font, fill=COLOR_WHITE_RGB)
        y += line_h + layout.dim(4)

    return _pil_to_bgr(canvas)


def _draw_candidate_tile(
    draw,
    img,
    layout: RenderLayout,
    *,
    x: int,
    y: int,
    width: int,
    candidate: dict,
    rank: int,
    selected: bool,
) -> int:
    thumb_h = layout.candidate_thumb_h
    label = str(candidate.get("page_label") or candidate.get("page_purpose") or candidate.get("node_id") or "")
    reasoning = str(candidate.get("vlm_reasoning") or "").strip()
    score = float(candidate.get("score") or 0.0)

    font_label = _load_font(layout.dim(12))
    font_small = _load_font(layout.dim(10))
    font_badge = _load_font(layout.dim(10), bold=True)
    inset = layout.dim(8)

    tile_h = thumb_h + layout.candidate_tile_text_h
    _draw_card(draw, x, y, x + width, y + tile_h, layout, selected=selected)

    shot = candidate.get("_screenshot_bgr")
    if shot is None:
        shot = np.full((thumb_h, width - 2 * inset, 3), 226, dtype=np.uint8)
    thumb, _, _ = _fit_image_contain_bgr(shot, width - 2 * inset, thumb_h)
    thumb_pil = _bgr_to_pil(thumb)
    img.paste(thumb_pil, (x + inset, y + inset))

    badge_y = y + layout.dim(12)
    badge_h = layout.dim(18)
    draw.rounded_rectangle(
        [x + layout.dim(12), badge_y, x + layout.dim(36), badge_y + badge_h],
        radius=layout.dim(8),
        fill=(60, 60, 60),
    )
    draw.text((x + layout.dim(16), badge_y + layout.dim(2)), f"#{rank}", font=font_badge, fill=COLOR_WHITE_RGB)

    score_text = f"{score:.3f}"
    score_w = draw.textlength(score_text, font=font_badge) + layout.dim(12)
    draw.rounded_rectangle(
        [x + layout.dim(42), badge_y, x + layout.dim(42) + score_w, badge_y + badge_h],
        radius=layout.dim(8),
        fill=COLOR_ACCENT_RGB,
    )
    draw.text((x + layout.dim(48), badge_y + layout.dim(2)), score_text, font=font_badge, fill=COLOR_WHITE_RGB)

    text_x = x + layout.dim(10)
    text_w = width - layout.dim(20)
    text_y = y + thumb_h + layout.dim(12)
    label_lines = _wrap_text(label, font_label, text_w, draw)[:2]
    for line in label_lines:
        draw.text((text_x, text_y), line, font=font_label, fill=COLOR_TEXT_RGB)
        text_y += _line_height(font_label, draw) + layout.dim(2)

    if reasoning:
        reason_lines = _wrap_text(reasoning, font_small, text_w, draw)[:2]
        for line in reason_lines:
            draw.text((text_x, text_y), line, font=font_small, fill=COLOR_MUTED_RGB)
            text_y += _line_height(font_small, draw) + 1

    if selected:
        draw.text(
            (text_x, y + tile_h - layout.dim(18)),
            "Selected",
            font=font_badge,
            fill=COLOR_ACCENT_RGB,
        )

    return tile_h


def render_candidates_frame(
    candidates: list[dict],
    selected_node_id: Optional[str],
    logs_root: str,
    *,
    layout: Optional[RenderLayout] = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    layout = layout or get_render_layout()
    canvas_width = layout.canvas_width
    enriched: list[dict] = []
    for c in candidates:
        item = dict(c)
        item["_screenshot_bgr"] = _load_candidate_screenshot(logs_root, str(item.get("node_id", "")))
        enriched.append(item)

    count = len(enriched)
    body_font = _load_font(layout.dim(13))
    dummy = Image.new("RGB", (canvas_width, 10))
    draw = ImageDraw.Draw(dummy)

    inner_w = layout.inner_width
    title = f"Found {count} candidate pages. Tap the best match:"
    selected = next((c for c in enriched if c.get("node_id") == selected_node_id), None)
    if selected:
        selected_label = str(selected.get("page_label") or selected.get("page_purpose") or selected_node_id)
        subtitle = f"Selected: {selected_label}"
    else:
        subtitle = "No candidate selected — navigating without memory anchor."

    title_lines = _wrap_text(title, body_font, inner_w, draw)
    subtitle_lines = _wrap_text(subtitle, body_font, inner_w, draw)

    cols = 2 if count > 1 else 1
    gap = layout.dim(12)
    tile_w = (inner_w - gap * (cols - 1)) // cols
    rows = (count + cols - 1) // cols
    tile_h = layout.candidate_thumb_h + layout.candidate_tile_text_h
    grid_h = rows * tile_h + max(0, rows - 1) * gap

    header_h = (
        layout.dim(20)
        + len(title_lines) * (_line_height(body_font, draw) + layout.dim(4))
        + layout.dim(8)
        + len(subtitle_lines) * (_line_height(body_font, draw) + layout.dim(4))
        + layout.dim(12)
    )
    total_h = header_h + grid_h + layout.dim(56)

    canvas = Image.new("RGB", (canvas_width, total_h), COLOR_PAGE_BG_BGR[::-1])
    draw = ImageDraw.Draw(canvas)

    card_x0 = layout.card_margin_x
    card_y0 = layout.dim(16)
    card_x1 = canvas_width - layout.card_margin_x
    card_y1 = total_h - layout.dim(16)
    _draw_card(draw, card_x0, card_y0, card_x1, card_y1, layout)

    x = card_x0 + layout.card_padding
    y = card_y0 + layout.card_padding
    for line in title_lines:
        draw.text((x, y), line, font=body_font, fill=COLOR_TEXT_RGB)
        y += _line_height(body_font, draw) + layout.dim(4)
    y += layout.dim(4)
    for line in subtitle_lines:
        draw.text((x, y), line, font=body_font, fill=COLOR_ACCENT_RGB if selected else COLOR_MUTED_RGB)
        y += _line_height(body_font, draw) + layout.dim(4)
    y += layout.dim(8)

    for idx, candidate in enumerate(enriched):
        row = idx // cols
        col = idx % cols
        tile_x = card_x0 + layout.card_padding + col * (tile_w + gap)
        tile_y = y + row * (tile_h + gap)
        is_selected = candidate.get("node_id") == selected_node_id
        _draw_candidate_tile(
            draw,
            canvas,
            layout,
            x=tile_x,
            y=tile_y,
            width=tile_w,
            candidate=candidate,
            rank=idx + 1,
            selected=is_selected,
        )

    return _pil_to_bgr(canvas)


def render_web_step_frame(
    step_num: int,
    action: dict,
    annotated_bgr: np.ndarray,
    *,
    layout: Optional[RenderLayout] = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    layout = layout or get_render_layout()
    canvas_width = layout.canvas_width
    thought = str(action.get("thought", "")).strip()
    action_type = str(action.get("type", "")).strip() or "unknown"
    direction = str(action.get("direction", "")).strip().lower()
    timing = action.get("timing") or {}
    if action_type in ("scroll", "swipe") and direction in ("up", "down", "left", "right"):
        action_display = f"{action_type} ({direction})"
    else:
        action_display = action_type

    def _fmt_timing(value) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "—"

    timing_lines = [
        f"driver screenshot: {_fmt_timing(timing.get('driver_screenshot_s'))} sec",
        f"model prediction: {_fmt_timing(timing.get('model_prediction_s'))} sec",
        f"other processing: {_fmt_timing(timing.get('other_processing_s'))} sec",
    ]

    font_step = _load_font(layout.dim(12), bold=True)
    font_mono = _load_font(layout.dim(12), mono=True)
    font_body = _load_font(layout.dim(14))
    font_timing = _load_font(layout.dim(11), mono=True)

    inner_w = layout.inner_width
    shot_max_h = layout.step_shot_max_h

    dummy = Image.new("RGB", (canvas_width, 10))
    draw = ImageDraw.Draw(dummy)
    thought_lines = _wrap_text(thought, font_body, inner_w, draw)

    shot_box, shot_w, shot_h = _fit_image_contain_bgr(annotated_bgr, inner_w, shot_max_h)

    text_block_h = (
        _line_height(font_step, draw)
        + layout.dim(6)
        + _line_height(font_mono, draw)
        + layout.dim(10)
        + len(thought_lines) * (_line_height(font_body, draw) + layout.dim(4))
    )
    timing_block_h = layout.dim(12) + len(timing_lines) * (_line_height(font_timing, draw) + layout.dim(2))
    total_h = layout.dim(32) + text_block_h + layout.dim(12) + shot_h + timing_block_h + layout.dim(32)

    canvas = Image.new("RGB", (canvas_width, total_h), COLOR_PAGE_BG_BGR[::-1])
    draw = ImageDraw.Draw(canvas)

    card_x0 = layout.card_margin_x
    card_y0 = layout.dim(16)
    card_x1 = canvas_width - layout.card_margin_x
    card_y1 = total_h - layout.dim(16)
    _draw_card(draw, card_x0, card_y0, card_x1, card_y1, layout)

    x = card_x0 + layout.card_padding
    y = card_y0 + layout.card_padding

    draw.text((x, y), f"Step {step_num}", font=font_step, fill=COLOR_ACCENT_RGB)
    y += _line_height(font_step, draw) + layout.dim(6)
    draw.text((x, y), f"Action: {action_display}", font=font_mono, fill=COLOR_MUTED_RGB)
    y += _line_height(font_mono, draw) + layout.dim(10)

    for line in thought_lines:
        draw.text((x, y), line, font=font_body, fill=COLOR_TEXT_RGB)
        y += _line_height(font_body, draw) + layout.dim(4)

    y += layout.dim(12)
    shot_pil = _bgr_to_pil(shot_box)
    border = layout.dim(1)
    border_canvas = Image.new("RGB", (shot_w + 2 * border, shot_h + 2 * border), COLOR_BORDER_RGB)
    border_canvas.paste(shot_pil, (border, border))
    canvas.paste(border_canvas, (x, y))
    y += shot_h + layout.dim(12)

    for line in timing_lines:
        draw.text((x, y), line, font=font_timing, fill=COLOR_MUTED_RGB)
        y += _line_height(font_timing, draw) + layout.dim(2)

    return _pil_to_bgr(canvas)


def normalize_frame_sizes(frames: list[np.ndarray], layout: Optional[RenderLayout] = None) -> list[np.ndarray]:
    if not frames:
        return []
    layout = layout or get_render_layout()
    canvas_width = layout.canvas_width
    max_h = max(frame.shape[0] for frame in frames)
    normalized: list[np.ndarray] = []
    pad_color = np.array(COLOR_PAGE_BG_BGR, dtype=np.uint8)
    for frame in frames:
        h, w = frame.shape[:2]
        if w != canvas_width:
            new_h = max(1, int(round(h * canvas_width / max(w, 1))))
            frame = cv2.resize(frame, (canvas_width, new_h), interpolation=cv2.INTER_LANCZOS4)
            h = frame.shape[0]
        if h < max_h:
            pad = np.broadcast_to(pad_color, (max_h - h, w, 3)).copy()
            frame = np.vstack([frame, pad])
        normalized.append(frame)
    return normalized


def build_gui_trajectory_frames(
    screenshots: list,
    actions: list,
    *,
    final_screenshot=None,
    query: Optional[str] = None,
    candidates: Optional[list[dict]] = None,
    selected_node_id: Optional[str] = None,
    logs_root: Optional[str] = None,
    annotate_screenshots=None,
) -> list[np.ndarray]:
    layout = get_render_layout()
    frames: list[np.ndarray] = []

    if query:
        frames.append(render_query_frame(query, layout))

    if candidates:
        frames.append(
            render_candidates_frame(
                candidates,
                selected_node_id,
                logs_root or "",
                layout=layout,
            )
        )

    if not screenshots or not actions:
        return normalize_frame_sizes(frames, layout)

    paired_shots = screenshots[: len(actions)]
    annotate = annotate_screenshots or (lambda s, a: s)
    annotated = annotate(paired_shots, actions)

    visible_steps = 0
    for step_idx, (img, action) in enumerate(zip(annotated, actions), start=1):
        action_type = str(action.get("type", "")).strip().lower()
        if action_type in {"finished", "finish"}:
            continue
        visible_steps += 1
        frames.append(render_web_step_frame(step_idx, action, img, layout=layout))

    if final_screenshot is not None:
        frames.append(
            render_web_step_frame(
                visible_steps + 1,
                {"type": "finish", "thought": "Navigation complete"},
                final_screenshot,
                layout=layout,
            )
        )

    return normalize_frame_sizes(frames, layout)
