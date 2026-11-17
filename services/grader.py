import subprocess
from pathlib import Path

LUT_PATH = Path(__file__).parent.parent / "luts"

GRADE_MAP = {
    "warm":      "warm.cube",
    "cool":      "cool.cube",
    "vibrant":   "vibrant.cube",
    "cinematic": "cinematic.cube",
    "natural":   None,
}


def apply_color_grade(input_path: Path, output_path: Path, grade: str):
    # Fast, predictable creator grades.  These are intentionally FFmpeg-native
    # so they work even when a LUT bundle is not deployed with the worker.
    native_filters = {
        "vibrant": "eq=saturation=1.22:contrast=1.06:brightness=0.02",
        "cinematic": "eq=saturation=0.92:contrast=1.12,colorbalance=rs=0.03:bs=-0.03",
        "warm": "colorbalance=rs=0.05:gs=0.02:bs=-0.04",
    }
    if grade in native_filters:
        subprocess.run([
            "ffmpeg", "-y", "-nostdin", "-i", str(input_path), "-vf", native_filters[grade],
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-c:a", "copy",
            "-movflags", "+faststart", str(output_path)
        ], check=True, capture_output=True, timeout=180)
        return
    lut_file = GRADE_MAP.get(grade)

    if not lut_file:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    lut_path = LUT_PATH / lut_file
    if not lut_path.exists():
        import shutil
        shutil.copy2(input_path, output_path)
        return

    subprocess.run([
        "ffmpeg", "-y", "-nostdin",
        "-i", str(input_path),
        "-vf", f"lut3d='{lut_path}'",
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "veryfast",
        "-c:a", "copy",
        str(output_path)
    ], check=True, capture_output=True, timeout=180)
