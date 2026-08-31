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


def _transcribe_local(video_path: Path) -> list[dict]:
    try:
        import whisper
    except ImportError:
        return []

    model = whisper.load_model("base")
    result = model.transcribe(
        str(video_path),
        word_timestamps=True,
        verbose=False,
    )

    words = []
    for segment in result.get("segments", []):
        for word_data in segment.get("words", []):
            word = word_data.get("word", "").strip()
            if word:
                words.append({
                    "word":  word,
                    "start": word_data["start"],
                    "end":   word_data["end"],
                })
    return words


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
) -> None:
    """
    Burns animated captions into the video using FFmpeg drawtext.
    Shows WORDS_PER_GROUP words at a time — readable on mobile.
    Falls back to silent copy if drawtext is unavailable.
    """
    if not _has_drawtext():
        print("[captions] ffmpeg drawtext filter is unavailable; skipping caption burn-in and copying input video")
        import shutil
        shutil.copy2(input_path, output_path)
        return

    font      = _detect_font()
    font_size = config.get("font_size", 60)
    color     = config.get("text_color", "white")
    highlight = config.get("highlight_color", "#FFD700")

    # Transcribe locally with Whisper
    words  = _transcribe_local(input_path)
    groups = _group_words(words, WORDS_PER_GROUP)

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
        "-preset", "medium",
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