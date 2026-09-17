import json
import subprocess
from pathlib import Path


def crop_to_vertical(
    input_path: Path,
    output_path: Path,
    target_w: int = 1080,
    target_h: int = 1920,
):
    width, height = get_dimensions(input_path)

    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read dimensions from {input_path}")

    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target dimensions must be greater than zero")

    target_ratio = target_w / target_h
    source_ratio = width / height

    if source_ratio > target_ratio:
        crop_h = height
        crop_w = int(round(height * target_ratio))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        x_offset = (width - crop_w) // 2
        y_offset = 0
    else:
        crop_w = width
        crop_h = int(round(width / target_ratio))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        x_offset = 0
        y_offset = (height - crop_h) // 2

    crop_w = max(2, min(crop_w, width))
    crop_h = max(2, min(crop_h, height))
    x_offset = max(0, (width - crop_w) // 2)
    y_offset = max(0, (height - crop_h) // 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        (
            f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},"
            f"scale={target_w}:{target_h}:flags=lanczos,"
            "format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-profile:v",
        "high",
        "-level",
        "4.2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg crop failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"FFmpeg did not create cropped video: {output_path}"
        )

    if output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Cropped video is empty: {output_path}"
        )

    output_width, output_height = get_dimensions(output_path)

    if output_width != target_w or output_height != target_h:
        raise RuntimeError(
            f"Unexpected cropped dimensions: "
            f"{output_width}x{output_height}, "
            f"expected {target_w}x{target_h}"
        )


def get_dimensions(video_path: Path) -> tuple[int, int]:
    if not video_path.exists():
        return 0, 0

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return 0, 0

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0, 0

    streams = data.get("streams", [])

    if not streams:
        return 0, 0

    stream = streams[0]

    return (
        int(stream.get("width") or 0),
        int(stream.get("height") or 0),
    )
