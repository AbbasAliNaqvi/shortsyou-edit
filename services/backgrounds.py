import shutil
import subprocess
from pathlib import Path


def _run_ffmpeg(args: list[str]):
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout[-4000:]}\n"
            f"STDERR:\n{result.stderr[-8000:]}"
        )


def _apply_blur_background(input_path: Path, output_path: Path):
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(input_path),

        "-filter_complex",
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:20[bg];"
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        "format=yuv420p[v]",

        "-map", "[v]",
        "-map", "0:a?",

        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "22",
        "-preset", "veryfast",

        "-c:a", "aac",
        "-b:a", "128k",

        "-movflags", "+faststart",

        str(output_path),
    ])



def _apply_dark_gradient(input_path: Path, output_path: Path):
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=0x1a1a2e:s=1080x1920:r=30",
        "-i", str(input_path),

        "-filter_complex",
        "[1:v]scale=1080:-1[scaled];"
        "[0:v][scaled]overlay=(W-w)/2:(H-h)/2-200,"
        "format=yuv420p[v]",

        "-map", "[v]",
        "-map", "1:a?",

        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "22",
        "-preset", "veryfast",

        "-c:a", "aac",
        "-b:a", "128k",

        "-shortest",
        "-movflags", "+faststart",

        str(output_path),
    ])


def _no_change(input_path: Path, output_path: Path):
    shutil.copy2(input_path, output_path)


def _apply_solid_color(
    input_path: Path,
    output_path: Path,
    color: str = "0x000000",
):
    _run_ffmpeg([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s=1080x1920:r=30",
        "-i", str(input_path),

        "-filter_complex",
        "[1:v]scale=1080:-1[scaled];"
        "[0:v][scaled]overlay=(W-w)/2:(H-h)/2,"
        "format=yuv420p[v]",

        "-map", "[v]",
        "-map", "1:a?",

        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "22",
        "-preset", "veryfast",

        "-c:a", "aac",
        "-b:a", "128k",

        "-shortest",
        "-movflags", "+faststart",

        str(output_path),
    ])


BACKGROUND_MODES = {
    "blur": _apply_blur_background,
    "dark_gradient": _apply_dark_gradient,
    "original": _no_change,
    "brand_color": _apply_solid_color,
}


def apply_background(input_path: Path, output_path: Path, mode: str):
    fn = BACKGROUND_MODES.get(mode, _apply_blur_background)
    fn(input_path, output_path)
