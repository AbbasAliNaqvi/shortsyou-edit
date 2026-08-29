import subprocess
import json
from pathlib import Path


def remove_silences_and_fillers(input_path: Path, output_path: Path):
    """
    Removes silence gaps longer than 0.5 seconds.
    Uses FFmpeg silencedetect to find gaps, then builds a select filter.
    Result is a cleaner, tighter clip.
    """
    # Detect silence
    result = subprocess.run([
        "ffmpeg",
        "-i", str(input_path),
        "-af", "silencedetect=noise=-30dB:d=0.5",
        "-f", "null", "-"
    ], capture_output=True, text=True)

    stderr = result.stderr
    silence_periods = parse_silence(stderr)

    if not silence_periods:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    # Build FFmpeg atrim filter to cut silence
    keep_segments = invert_silence(silence_periods, get_duration(input_path))

    if not keep_segments:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    # Build complex filter
    video_parts = []
    audio_parts = []
    for i, (start, end) in enumerate(keep_segments):
        video_parts.append(f"[0:v]trim={start}:{end},setpts=PTS-STARTPTS[v{i}]")
        audio_parts.append(f"[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS[a{i}]")

    n = len(keep_segments)
    concat_v = "".join(f"[v{i}]" for i in range(n))
    concat_a = "".join(f"[a{i}]" for i in range(n))

    filter_complex = ";".join(video_parts + audio_parts)
    filter_complex += f";{concat_v}concat=n={n}:v=1:a=0[vout]"
    filter_complex += f";{concat_a}concat=n={n}:v=0:a=1[aout]"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-crf", "22",
        "-c:a", "aac",
        str(output_path)
    ], check=True, capture_output=True)


def parse_silence(stderr: str) -> list[tuple[float, float]]:
    import re
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", stderr)]
    return list(zip(starts, ends[:len(starts)]))


def invert_silence(silence_periods: list, duration: float) -> list[tuple[float, float]]:
    keep = []
    cursor = 0.0
    for start, end in silence_periods:
        if start > cursor:
            keep.append((cursor, start))
        cursor = end
    if cursor < duration:
        keep.append((cursor, duration))
    return keep


def get_duration(path: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(path)
    ], capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"].get("duration", 0))