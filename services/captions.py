import json
import subprocess
import tempfile
from pathlib import Path


FONT_FALLBACK_ORDER = ["Impact", "Anton", "Arial-Bold", "DejaVu-Sans-Bold"]
WORDS_PER_GROUP = 4


def _detect_font() -> str:
    result = subprocess.run(
        ["fc-list"],
        capture_output=True,
        text=True,
    )
    installed = result.stdout.lower()
    for font in FONT_FALLBACK_ORDER:
        if font.lower().replace("-", " ") in installed:
            return font
    return "monospace"


def _has_drawtext() -> bool:
    result = subprocess.run(
        ["ffmpeg", "-filters"],
        capture_output=True,
        text=True,
    )
    return "drawtext" in result.stdout


def _group_words(words: list[dict], group_size: int = WORDS_PER_GROUP) -> list[dict]:
    """
    Groups individual word timestamps into caption phrases.
    Showing 3-4 words at a time is more readable than one word at a time.
    """
    groups = []
    for i in range(0, len(words), group_size):
        chunk = words[i : i + group_size]
        if not chunk:
            continue
        groups.append({
            "text":  " ".join(w["word"] for w in chunk),
            "start": chunk[0]["start"],
            "end":   chunk[-1]["end"],
        })
    return groups


def _escape(text: str) -> str:
    return (
        text
        .replace("\\", "\\\\")
        .replace("'",  "\u2019")   # smart apostrophe — avoids shell quoting issues
        .replace(":",  "\\:")
        .replace("%",  "\\%")
        .replace("[",  "\\[")
        .replace("]",  "\\]")
    )


def add_captions(
    input_path: Path,
    output_path: Path,
    config: dict,
    hook_text: str = None,
    caption_words: list[dict] | None = None,
) -> None:
    """
    Burns caption words supplied by the transcription service. This service
    deliberately never performs ASR/Whisper: exporting must remain fast and
    use the authoritative timing already saved by the API.
    """
    font      = _detect_font()
    font_size = config.get("font_size", 60)
    color     = config.get("text_color", "white")
    highlight = config.get("highlight_color", "#FFD700")

    words = caption_words or []
    if caption_words:
        print(f"[captions] source=transcription_service words={len(caption_words)}")
    else:
        print("[captions] source=none; rendering without spoken captions")
    groups = _group_words(words, WORDS_PER_GROUP)

    # Some local FFmpeg builds (including the installed one on this machine)
    # omit libfreetype and therefore drawtext. Do not silently ship an
    # uncaptioned short: OpenCV provides a portable fallback.
    if not _has_drawtext():
        print("[captions] ffmpeg drawtext unavailable; using OpenCV caption renderer")
        _burn_captions_with_opencv(input_path, output_path, groups, hook_text, color, highlight)
        return

    filters = []

    # Hook text — top of screen, first 3 seconds, yellow
    if hook_text:
        escaped = _escape(hook_text)
        filters.append(
            f"drawtext="
            f"text='{escaped}':"
            f"fontfile=/System/Library/Fonts/Supplemental/Impact.ttf:"
            f"fontsize={font_size - 6}:"
            f"fontcolor={highlight}:"
            f"x=(w-text_w)/2:"
            f"y=120:"
            f"enable='between(t,0,3)':"
            f"box=1:"
            f"boxcolor=black@0.55:"
            f"boxborderw=10"
        )

    # Caption groups — bottom third, white with dark box
    for group in groups:
        escaped = _escape(group["text"])
        start   = group["start"]
        end     = group["end"]

        filters.append(
            f"drawtext="
            f"text='{escaped}':"
            f"fontsize={font_size}:"
            f"fontcolor={color}:"
            f"x=(w-text_w)/2:"
            f"y=h-220:"
            f"enable='between(t,{start:.3f},{end:.3f})':"
            f"box=1:"
            f"boxcolor=black@0.6:"
            f"boxborderw=12"
        )

    # If no transcription, at least show the hook
    if not filters:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "veryfast",
        "-c:a", "copy",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[captions] ffmpeg caption burn failed:\n{result.stderr}")
        import shutil
        shutil.copy2(input_path, output_path)
        return

    if not output_path.exists() or output_path.stat().st_size == 0:
        import shutil
        shutil.copy2(input_path, output_path)


def _burn_captions_with_opencv(
    input_path: Path,
    output_path: Path,
    groups: list[dict],
    hook_text: str | None,
    color: str,
    highlight: str,
) -> None:
    """Portable caption path for FFmpeg installations without drawtext."""
    try:
        import cv2
    except ImportError:
        import shutil
        print("[captions] OpenCV unavailable; copying uncaptioned video")
        shutil.copy2(input_path, output_path)
        return

    def parse_color(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        try:
            return tuple(int(value[i:i + 2], 16) for i in (4, 2, 0))
        except ValueError:
            return (255, 255, 255)

    def draw_text(frame, text: str, y: int, text_color: tuple[int, int, int]):
        lines = [text[i:i + 27] for i in range(0, len(text), 27)] or [text]
        font, scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 1.25, 3
        line_height = 50
        top = y - 42
        bottom = y + line_height * len(lines) + 12
        cv2.rectangle(frame, (32, top), (frame.shape[1] - 32, bottom), (0, 0, 0), -1)
        for index, line in enumerate(lines):
            (width, _), _ = cv2.getTextSize(line, font, scale, thickness)
            x = max(20, (frame.shape[1] - width) // 2)
            cv2.putText(frame, line, (x, y + index * line_height), font, scale, text_color, thickness, cv2.LINE_AA)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or width <= 0 or height <= 0:
        cap.release()
        import shutil
        shutil.copy2(input_path, output_path)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        silent_path = Path(temp_dir) / "captioned-silent.mp4"
        writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            import shutil
            shutil.copy2(input_path, output_path)
            return
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if hook_text and timestamp <= 3:
                draw_text(frame, hook_text, 120, parse_color(highlight))
            for group in groups:
                if group["start"] <= timestamp <= group["end"]:
                    draw_text(frame, group["text"], height - 180, parse_color(color))
                    break
            writer.write(frame)
        writer.release()
        cap.release()
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(silent_path), "-i", str(input_path),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(output_path),
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[captions] OpenCV caption mux failed:\n{result.stderr}")
            import shutil
            shutil.copy2(input_path, output_path)
