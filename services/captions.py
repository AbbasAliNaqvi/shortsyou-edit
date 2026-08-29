import subprocess
from pathlib import Path
import whisper


def _ffmpeg_supports_filter(filter_name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=True,
    )
    return f" {filter_name} " in result.stdout


def _font_exists(name: str) -> bool:
    try:
        result = subprocess.run(
            ["fc-list", name],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        return False


def _get_font_file() -> str:
    """
    Prefer system Impact.
    Fall back to the project's local Anton font.
    """
    if _font_exists("Impact"):
        return "Impact"

    project_root = Path(__file__).resolve().parent.parent
    anton_path = project_root / "fonts" / "Anton.woff2"

    if anton_path.exists():
        return str(anton_path)

    return "Anton"


def _escape_drawtext(text: str) -> str:
    """Escape text for FFmpeg drawtext."""
    return (
        text.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("%", r"\%")
        .replace("[", r"\[")
        .replace("]", r"\]")
    )


async def add_captions(
    input_path: Path,
    output_path: Path,
    config: dict,
    hook_text: str = None,
):
    """
    Transcribes video with Whisper (runs locally, free).
    Adds word-by-word animated captions using FFmpeg drawtext.

    Uses Impact when installed, otherwise falls back to
    the project's local Anton.woff2 font.

    If hook_text is provided, it appears at the top
    for the first 3 seconds.
    """

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _ffmpeg_supports_filter("drawtext"):
        print(
            "[captions] ffmpeg drawtext filter is unavailable; "
            "skipping caption burn-in and copying input video"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return

    # ---------------------------------------------------------
    # Select font
    # ---------------------------------------------------------
    font_file = _get_font_file()

    # ---------------------------------------------------------
    # Transcribe video with Whisper
    # ---------------------------------------------------------
    model = whisper.load_model("base")

    result = model.transcribe(
        str(input_path),
        word_timestamps=True,
        verbose=False,
    )

    # ---------------------------------------------------------
    # Build FFmpeg filters
    # ---------------------------------------------------------
    filters = []

    # ---------------------------------------------------------
    # Hook text
    # ---------------------------------------------------------
    if hook_text:
        escaped = _escape_drawtext(hook_text)

        filters.append(
            "drawtext="
            f"text='{escaped}':"
            "fontsize=48:"
            "fontcolor=yellow:"
            "x=(w-text_w)/2:"
            "y=80:"
            "enable='between(t,0,3)':"
            "box=1:"
            "boxcolor=black@0.5:"
            "boxborderw=8:"
            f"fontfile='{font_file}'"
        )

    # ---------------------------------------------------------
    # Word-by-word captions
    # ---------------------------------------------------------
    font_size = config.get("font_size", 52)
    color = config.get("text_color", "white")
    highlight = config.get("highlight_color", "#FFD700")

    for segment in result.get("segments", []):
        for word_data in segment.get("words", []):
            word = word_data.get("word", "").strip()

            if not word:
                continue

            start = word_data.get("start")
            end = word_data.get("end")

            if start is None or end is None:
                continue

            escaped = _escape_drawtext(word)

            # -------------------------------------------------
            # Normal caption
            # -------------------------------------------------
            filters.append(
                "drawtext="
                f"text='{escaped}':"
                f"fontsize={font_size}:"
                f"fontcolor={color}:"
                "x=(w-text_w)/2:"
                "y=h-200:"
                f"enable='between(t,{start},{end})':"
                "box=1:"
                "boxcolor=black@0.4:"
                "boxborderw=6:"
                f"fontfile='{font_file}'"
            )

            # -------------------------------------------------
            # Highlighted current word
            # -------------------------------------------------
            filters.append(
                "drawtext="
                f"text='{escaped}':"
                f"fontsize={font_size + 4}:"
                f"fontcolor={highlight}:"
                "x=(w-text_w)/2:"
                "y=h-200:"
                f"enable='between(t,{start},{end})':"
                f"fontfile='{font_file}'"
            )

    # ---------------------------------------------------------
    # Apply FFmpeg
    # ---------------------------------------------------------
    filter_str = ",".join(filters) if filters else "null"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            filter_str,
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-preset",
            "medium",
            "-c:a",
            "copy",
            str(output_path),
        ],
        check=True,
    )