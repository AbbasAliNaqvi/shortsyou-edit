import shutil
import subprocess
from pathlib import Path


def _apply_blur_background(input_path: Path, output_path: Path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:20[bg];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ], check=True, capture_output=True)


def _apply_dark_gradient(input_path: Path, output_path: Path):
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x1a1a2e:s=1080x1920:r=30",
        "-i", str(input_path),
        "-filter_complex",
        "[1:v]scale=1080:-1[scaled];"
        "[0:v][scaled]overlay=(W-w)/2:(H-h)/2-200",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-map", "1:a", "-c:a", "aac", "-shortest",
        str(output_path),
    ], check=True, capture_output=True)


def _no_change(input_path: Path, output_path: Path):
    shutil.copy2(input_path, output_path)


def _apply_solid_color(input_path: Path, output_path: Path, color: str = "0x000000"):
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:s=1080x1920:r=30",
        "-i", str(input_path),
        "-filter_complex",
        "[1:v]scale=1080:-1[scaled];"
        "[0:v][scaled]overlay=(W-w)/2:(H-h)/2",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-map", "1:a", "-c:a", "aac", "-shortest",
        str(output_path),
    ], check=True, capture_output=True)


# This dict must be at module level, not inside any function
BACKGROUND_MODES = {
    "blur":          _apply_blur_background,
    "dark_gradient": _apply_dark_gradient,
    "original":      _no_change,
    "brand_color":   _apply_solid_color,
}


def apply_background(input_path: Path, output_path: Path, mode: str):
    fn = BACKGROUND_MODES.get(mode, _apply_blur_background)
    fn(input_path, output_path)