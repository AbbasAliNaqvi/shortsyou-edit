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
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"lut3d='{lut_path}'",
        "-c:v", "libx264",
        "-crf", "22",
        "-preset", "medium",
        "-c:a", "copy",
        str(output_path)
    ], check=True, capture_output=True)