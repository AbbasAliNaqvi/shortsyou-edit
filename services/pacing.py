"""Deterministic FFmpeg digital punch-ins from an approved director plan."""

import shutil
import subprocess
from pathlib import Path


def apply_camera_punch_ins(input_path: Path, output_path: Path, duration: float, cues: list[dict]) -> None:
    cues = sorted((c for c in cues if isinstance(c, dict) and 0 < float(c.get("timestamp", 0)) < duration), key=lambda c: float(c["timestamp"]))
    if not cues:
        shutil.copy2(input_path, output_path)
        return
    points = [0.0] + [float(c["timestamp"]) for c in cues] + [duration]
    scales, current = [], 1.0
    for index in range(len(points) - 1):
        if index:
            current = min(1.15, max(1.0, float(cues[index - 1].get("scale", 1.12))))
        scales.append(current)
    chains, concat_inputs = [], []
    for i, (start, end, scale) in enumerate(zip(points, points[1:], scales)):
        vlabel, alabel = f"v{i}", f"a{i}"
        # Crop around the center after enlargement, preserving the speaker-safe
        # vertical frame while making an editorially clean hard cut.
        chains.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,scale=iw*{scale}:ih*{scale},crop=iw/{scale}:ih/{scale}:(iw-iw/{scale})/2:(ih-ih/{scale})/2,scale=1080:1920:flags=lanczos,setsar=1[{vlabel}]")
        chains.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[{alabel}]")
        # concat expects each segment's video/audio pair before the next pair.
        concat_inputs.extend((f"[{vlabel}]", f"[{alabel}]"))
    chains.append("".join(concat_inputs) + f"concat=n={len(scales)}:v=1:a=1[vout][aout]")
    result = subprocess.run(["ffmpeg", "-y", "-nostdin", "-i", str(input_path), "-filter_complex", ";".join(chains), "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output_path)], capture_output=True, text=True, timeout=180)
    if result.returncode:
        shutil.copy2(input_path, output_path)
