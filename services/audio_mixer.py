"""Dialogue-first audio mastering and sidechain music ducking."""

import math
import random
import shutil
import subprocess
from pathlib import Path

from .music import MOOD_MAP, MUSIC_LIBRARY_PATH


def mix_mastered_audio(input_path: Path, output_path: Path, mood: str, base_volume_db: float = -22, duck_during_speech_db: float = -32) -> None:
    folder = MUSIC_LIBRARY_PATH / MOOD_MAP.get(mood, "neutral")
    tracks = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
    if not tracks:
        normalize_dialogue(input_path, output_path)
        return
    music = random.choice(tracks)
    volume = math.pow(10, float(base_volume_db) / 20)
    duck_depth = abs(float(duck_during_speech_db) - float(base_volume_db))
    ratio = min(16, max(4, round(duck_depth)))
    # sidechaincompress reduces music while dialogue is present; the lower target
    # volume is represented by a conservative threshold/ratio instead of clipping.
    filters = (
        "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,asplit=2[voice][sidechain];"
        f"[1:a]volume={volume:.6f},aresample=48000,atrim=duration=99999,afade=t=in:st=0:d=.35[music];"
        # The compressor is level-dependent: music falls behind speech then
        # naturally returns to its base level during pauses.
        f"[music][sidechain]sidechaincompress=threshold=0.025:ratio={ratio}:attack=15:release=350[ducked];"
        "[voice][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.944[aout]"
    )
    cmd = ["ffmpeg", "-y", "-nostdin", "-i", str(input_path), "-stream_loop", "-1", "-i", str(music), "-filter_complex", filters,
           "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode:
        normalize_dialogue(input_path, output_path)


def normalize_dialogue(input_path: Path, output_path: Path) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-nostdin", "-i", str(input_path), "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-movflags", "+faststart", str(output_path)], capture_output=True, text=True, timeout=180)
    if result.returncode:
        shutil.copy2(input_path, output_path)
