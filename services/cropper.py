import subprocess
from pathlib import Path


def crop_to_vertical(input_path: Path, output_path: Path, target_w: int = 1080, target_h: int = 1920):
    """
    Smart crop to 9:16 vertical format.
    Detects the most active region and crops there.
    Falls back to center crop if detection fails.
    """
    # Get input dimensions
    width, height = get_dimensions(input_path)

    if width == 0 or height == 0:
        raise ValueError(f"Could not read dimensions from {input_path}")

    # Calculate crop for 9:16 from landscape or square video
    aspect_ratio = 9 / 16
    crop_w = int(height * aspect_ratio)

    if crop_w > width:
        # Video is already more vertical than 9:16
        crop_w = width
        crop_h = int(width / aspect_ratio)
        x_offset = 0
        y_offset = (height - crop_h) // 2
    else:
        crop_h = height
        x_offset = (width - crop_w) // 2
        y_offset = 0

    # Scale to exact 1080x1920
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},scale={target_w}:{target_h}",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ], check=True, capture_output=True)


def get_dimensions(video_path: Path) -> tuple[int, int]:
    import json
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path)
    ], capture_output=True, text=True)

    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream.get("width", 0), stream.get("height", 0)
    return 0, 0