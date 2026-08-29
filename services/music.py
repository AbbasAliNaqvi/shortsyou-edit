import os
import random
import subprocess
from pathlib import Path

MUSIC_LIBRARY_PATH = Path(__file__).parent.parent / "music_library"

MOOD_MAP = {
    "energetic": "energetic",
    "calm":      "calm",
    "excited":   "energetic",
    "sad":       "calm",
    "angry":     "dramatic",
    "happy":     "motivational",
    "neutral":   "neutral",
}


def select_and_mix_music(input_path: Path, output_path: Path, mood: str, music_volume: float = 0.15):
    """
    Selects a random track from the mood folder.
    Mixes it under the voice at -18dB (very subtle).
    Fades in over 0.5s, fades out over 1.0s.
    """
    folder = MUSIC_LIBRARY_PATH / MOOD_MAP.get(mood, "neutral")
    tracks = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))

    if not tracks:
        # No music available, just copy
        import shutil
        shutil.copy2(input_path, output_path)
        return

    track = random.choice(tracks)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", str(track),
        "-filter_complex",
        f"[1:a]volume={music_volume},afade=t=in:st=0:d=0.5,afade=t=out:d=1[music];"
        f"[0:a][music]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path)
    ], check=True, capture_output=True)