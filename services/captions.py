"""Professional captions, subtitle, and viral hook rendering engine.

Burns word-level transcription timestamps from the ML service or Groq Whisper
onto short-form vertical video with active karaoke highlighting, high-contrast
typography, safe-zone placement, and aesthetic top hook headlines (CapCut / OpusClip style).
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SYSTEM_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/Impact.ttf",
]


def _detect_script(text: str) -> str:
    """Detect writing script of the text."""
    for ch in text:
        cp = ord(ch)
        if 0x0A00 <= cp <= 0x0A7F:
            return "gurmukhi"   # Punjabi
        if 0x0900 <= cp <= 0x097F:
            return "devanagari" # Hindi / Marathi
        if 0x0600 <= cp <= 0x06FF:
            return "arabic"     # Urdu / Arabic
        if 0x0980 <= cp <= 0x09FF:
            return "bengali"
        if 0x0A80 <= cp <= 0x0AFF:
            return "gujarati"
        if 0x0B80 <= cp <= 0x0BFF:
            return "tamil"
        if 0x0C00 <= cp <= 0x0C7F:
            return "telugu"
    return "latin"


SCRIPT_FONT_CANDIDATES = {
    "gurmukhi": [
        "/System/Library/Fonts/Supplemental/Gurmukhi Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Gurmukhi MN.ttc",
        "/System/Library/Fonts/Supplemental/Gurmukhi.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
    "devanagari": [
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/ITFDevanagari.ttc",
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
        "/System/Library/Fonts/Kohinoor.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
    "universal": [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
}


def _find_font(font_size: int, sample_text: str = "") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    script = _detect_script(sample_text)
    candidates = []
    if script in SCRIPT_FONT_CANDIDATES:
        candidates.extend(SCRIPT_FONT_CANDIDATES[script])
    elif any(ord(c) > 127 for c in sample_text):
        candidates.extend(SCRIPT_FONT_CANDIDATES["universal"])

    candidates.extend(SYSTEM_FONT_CANDIDATES)

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue

    # Search local fonts directory
    local_fonts = Path(__file__).parent.parent / "fonts"
    if local_fonts.is_dir():
        for f in local_fonts.glob("*.ttf"):
            try:
                return ImageFont.truetype(str(f), font_size)
            except Exception:
                continue

    try:
        return ImageFont.truetype("Arial", font_size)
    except Exception:
        return ImageFont.load_default()


@functools.lru_cache(maxsize=32)
def get_script_font(script: str, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Cached font provider mapped specifically to the required Unicode writing script."""
    sample_map = {
        "gurmukhi": "ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ",
        "devanagari": "नमस्ते दुनिया",
        "arabic": "مرحبا",
        "bengali": "নমস্কার",
        "gujarati": "નમસ્તે",
        "tamil": "வணக்கம்",
        "telugu": "నమస్కారం",
    }
    sample = sample_map.get(script, "VIRAL SHORT")
    return _find_font(font_size, sample)


def _parse_color(value: str | None, default: tuple[int, int, int] = (255, 255, 255)) -> tuple[int, int, int]:
    if not value:
        return default
    val = value.strip().lstrip("#")
    if len(val) == 6:
        try:
            return tuple(int(val[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    val_lower = value.lower()
    if val_lower == "yellow":
        return (255, 230, 0)
    if val_lower == "white":
        return (255, 255, 255)
    if val_lower == "cyan":
        return (0, 255, 255)
    if val_lower == "green":
        return (0, 255, 102)
    if val_lower == "coral":
        return (255, 51, 102)
    return default


def _group_words(words: list[dict], group_size: int = 3) -> list[dict]:
    """Groups 2 to 3 words into punchy subtitle phrases for vertical reading.
    Extends word groups slightly over small silences (<0.25s) to eliminate flicker.
    """
    if not words:
        return []

    groups = []
    clean_words = []
    for w in words:
        txt = str(w.get("word", "")).strip()
        if not txt:
            continue
        clean_words.append({
            "word": txt.upper(),
            "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0)),
        })

    for i in range(0, len(clean_words), group_size):
        chunk = clean_words[i : i + group_size]
        if not chunk:
            continue

        g_start = chunk[0]["start"]
        g_end = chunk[-1]["end"]
        # Small minimum display duration for short words
        if g_end - g_start < 0.4:
            g_end = g_start + 0.4

        groups.append({
            "words": chunk,
            "start": g_start,
            "end": g_end,
            "text": " ".join(c["word"] for c in chunk),
        })

    # Smooth adjacent gaps under 0.25s
    for i in range(len(groups) - 1):
        gap = groups[i + 1]["start"] - groups[i]["end"]
        if 0 < gap < 0.25:
            groups[i]["end"] = groups[i + 1]["start"]

    return groups


def add_captions(
    input_path: Path,
    output_path: Path,
    config: dict,
    hook_text: str | None = None,
    caption_words: list[dict] | None = None,
    director_plan: dict | None = None,
) -> None:
    """Burns authoritative word timestamps onto the short video.
    Features:
    - Sleek viral hook pill badge during opening 3.2s
    - Active karaoke word highlighting in high-contrast neon/gold
    - Shadow drop and thick stroke for 100% legibility on any background
    - Direct streaming pipe into FFmpeg for maximum quality and speed
    """
    words = caption_words or []
    director_plan = director_plan or {}
    hook_headline = str(director_plan.get("hook_headline") or hook_text or "").strip().upper()

    has_words = len(words) > 0
    if not has_words and not hook_headline:
        print("[captions] No word timestamps or hook to render; preserving video source")
        shutil.copy2(input_path, output_path)
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[captions] Failed to open video: {input_path}")
        shutil.copy2(input_path, output_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        shutil.copy2(input_path, output_path)
        return

    print(
        f"[captions] Rendering studio-grade captions | resolution={width}x{height} "
        f"fps={fps:.2f} frames={total_frames} words={len(words)} hook={bool(hook_headline)}"
    )

    # Typography sizing based on frame dimensions (scales dynamically for 720p or 1080p)
    font_size = max(28, int(height * 0.038))
    stroke_w = max(3, int(font_size * 0.09))

    # Comprehensive script detection across ALL spoken words & viral headline
    all_words_text = " ".join(str(w.get("word", "")) for w in words)
    video_script = _detect_script(all_words_text)
    if video_script == "latin" and hook_headline:
        video_script = _detect_script(hook_headline)

    base_font = get_script_font(video_script, font_size)

    text_color = _parse_color(config.get("text_color"), (255, 255, 255))
    highlight_color = _parse_color(config.get("highlight_color"), (255, 230, 0))

    groups = _group_words(words, group_size=3)
    phrase_map = {
        str(item.get("word", "")).lower().strip(".,!?;:'\""): item
        for item in director_plan.get("key_phrases", [])
        if isinstance(item, dict)
    }
    safe_zone_y = float(director_plan.get("safe_zone_y", 72))

    space_bbox = base_font.getbbox(" ")
    space_w = space_bbox[2] - space_bbox[0]

    # Direct streaming pipe to FFmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-af",
        "loudnorm=I=-14:LRA=11:TP=-1.5",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            frame_idx += 1

            if total_frames > 0 and frame_idx % max(1, total_frames // 4) == 0:
                pct = int((frame_idx / total_frames) * 100)
                print(f"[captions] burning captions progress: {pct}% ({frame_idx}/{total_frames} frames)")

            active_group = None
            for g in groups:
                if g["start"] <= timestamp <= g["end"]:
                    active_group = g
                    break

            is_hook_active = bool(hook_headline and timestamp <= 3.2)

            # Fast path: frame has neither subtitles nor hook card
            if active_group is None and not is_hook_active:
                proc.stdin.write(frame.tobytes())
                continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)

            # 1. Top Viral Hook Badge (first 3.2 seconds with soft fade-out)
            if is_hook_active:
                alpha = 240 if timestamp < 2.7 else int(240 * max(0.0, (3.2 - timestamp) / 0.5))
                if alpha > 10:
                    badge_font_size = max(22, int(height * 0.024))
                    badge_script = _detect_script(hook_headline)
                    badge_font = get_script_font(badge_script if badge_script != "latin" else video_script, badge_font_size)
                    b_box = badge_font.getbbox(hook_headline)
                    bw = b_box[2] - b_box[0]
                    bh = b_box[3] - b_box[1]

                    pad_x = 24
                    pad_y = 10
                    card_w = min(width - 48, bw + pad_x * 2)
                    card_h = bh + pad_y * 2
                    card_x1 = (width - card_w) // 2
                    card_y1 = int(height * 0.12)
                    card_x2 = card_x1 + card_w
                    card_y2 = card_y1 + card_h

                    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    ov_draw = ImageDraw.Draw(overlay)
                    ov_draw.rounded_rectangle(
                        [card_x1, card_y1, card_x2, card_y2],
                        radius=card_h // 2,
                        fill=(12, 14, 20, alpha),
                        outline=(255, 230, 0, alpha),
                        width=2,
                    )
                    text_x = card_x1 + (card_w - bw) // 2
                    text_y = card_y1 + (card_h - bh) // 2 - b_box[1]
                    ov_draw.text(
                        (text_x, text_y),
                        hook_headline,
                        font=badge_font,
                        fill=(255, 255, 255, alpha),
                        stroke_width=2,
                        stroke_fill=(0, 0, 0, alpha),
                    )
                    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                    draw = ImageDraw.Draw(img)

            # 2. Spoken Karaoke Subtitles
            if active_group:
                g_words = active_group["words"]
                word_fonts = [
                    get_script_font(
                        _detect_script(w["word"]) if _detect_script(w["word"]) != "latin" else video_script,
                        font_size,
                    )
                    for w in g_words
                ]
                word_bboxes = [word_fonts[i].getbbox(w["word"]) for i, w in enumerate(g_words)]
                word_widths = [b[2] - b[0] for b in word_bboxes]
                word_heights = [b[3] - b[1] for b in word_bboxes]

                total_w = sum(word_widths) + space_w * (len(g_words) - 1)
                start_x = (width - total_w) // 2
                start_y = int(height * (safe_zone_y / 100))

                # Deep drop shadow for ultimate legibility
                cur_x = start_x
                for i, w_item in enumerate(g_words):
                    w_text = w_item["word"]
                    w_w = word_widths[i]
                    w_font = word_fonts[i]
                    draw.text(
                        (cur_x + 3, start_y + 4),
                        w_text,
                        font=w_font,
                        fill=(0, 0, 0, 220),
                        stroke_width=stroke_w + 1,
                        stroke_fill=(0, 0, 0, 220),
                    )
                    cur_x += w_w + space_w

                # Karaoke active word highlighting
                cur_x = start_x
                for i, w_item in enumerate(g_words):
                    w_text = w_item["word"]
                    w_w = word_widths[i]
                    w_font = word_fonts[i]
                    is_active = w_item["start"] <= timestamp <= w_item["end"]
                    token = w_text.lower().strip(".,!?;:'\"")
                    color = (
                        highlight_color
                        if is_active
                        else _parse_color(phrase_map.get(token, {}).get("color"), text_color)
                    )
                    emoji = str(phrase_map.get(token, {}).get("emoji", "")).strip()

                    draw.text(
                        (cur_x, start_y),
                        w_text,
                        font=w_font,
                        fill=color,
                        stroke_width=stroke_w,
                        stroke_fill=(0, 0, 0),
                    )
                    if emoji:
                        try:
                            draw.text(
                                (cur_x + w_w + 4, start_y - max(6, font_size // 4)),
                                emoji,
                                font=w_font,
                                fill=(255, 255, 255),
                            )
                        except Exception:
                            pass
                    cur_x += w_w + space_w

            bgr_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            proc.stdin.write(bgr_frame.tobytes())

        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read()
        proc.wait()

        if proc.returncode != 0:
            print(f"[captions] FFmpeg pipe error: {stderr[-2000:]}")
            shutil.copy2(input_path, output_path)
            return

        print(
            f"[captions] Direct pipe caption burn complete! Output: {output_path} "
            f"({output_path.stat().st_size} bytes)"
        )
    except Exception as e:
        cap.release()
        try:
            proc.kill()
        except Exception:
            pass
        print(f"[captions] Exception during caption pipe: {e}")
        shutil.copy2(input_path, output_path)
