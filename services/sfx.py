import json
import random
import shutil
import subprocess
from pathlib import Path

SFX_ROOT = Path(__file__).parent.parent / "sfx_library"

SFX_MAP = {
    "laugh":       SFX_ROOT / "reactions"    / "laugh_crowd.mp3",
    "laugh_light": SFX_ROOT / "reactions"    / "laugh_light.mp3",
    "love":        SFX_ROOT / "reactions"    / "love_chime.mp3",
    "shocked":     SFX_ROOT / "reactions"    / "shocked_gasp.mp3",
    "applause":    SFX_ROOT / "reactions"    / "applause_short.mp3",
    "whoosh":      SFX_ROOT / "transitions"  / "whoosh_fast.mp3",
    "whoosh_slow": SFX_ROOT / "transitions"  / "whoosh_slow.mp3",
    "impact":      SFX_ROOT / "transitions"  / "impact_hit.mp3",
    "riser":       SFX_ROOT / "cinematic"    / "riser_tension.mp3",
    "bass_drop":   SFX_ROOT / "cinematic"    / "bass_drop.mp3",
    "pop":         SFX_ROOT / "alerts"       / "pop_notification.mp3",
    "ding":        SFX_ROOT / "alerts"       / "ding_success.mp3",
}

DEFAULT_SFX_VOLUME = 0.55


def add_sfx(
    input_path: Path,
    output_path: Path,
    events: list[dict],
    sfx_volume: float = DEFAULT_SFX_VOLUME,
) -> None:
    """
    Layers sound effects onto a video at specific timestamps.

    events format:
    [
        { "type": "whoosh", "at_second": 0.0 },
        { "type": "laugh",  "at_second": 12.4 },
        { "type": "love",   "at_second": 28.0 }
    ]
    """
    if not events:
        shutil.copy2(input_path, output_path)
        return

    # Resolve which SFX files exist
    resolved = []
    for event in events:
        sfx_type = event.get("type", "whoosh")
        sfx_path = SFX_MAP.get(sfx_type)
        if sfx_path and sfx_path.exists():
            resolved.append({
                "type":      sfx_type,
                "path":      sfx_path,
                "at_second": float(event.get("at_second", 0)),
            })
        else:
            print(f"[sfx] skipping '{sfx_type}' — file not found: {sfx_path}")

    if not resolved:
        shutil.copy2(input_path, output_path)
        return

    # Build FFmpeg inputs list
    # Input 0 = main video, Inputs 1..N = SFX files
    cmd = ["-y", "-i", str(input_path)]

    seen_paths = {}
    sfx_input_indices = []

    for item in resolved:
        path_str = str(item["path"])
        if path_str not in seen_paths:
            seen_paths[path_str] = len(cmd) // 2  # next -i index
            cmd += ["-i", path_str]
        sfx_input_indices.append(seen_paths[path_str])

    # Build filter_complex
    delay_filters = []
    mix_labels    = ["[0:a]"]

    for i, item in enumerate(resolved):
        idx       = sfx_input_indices[i]
        delay_ms  = int(item["at_second"] * 1000)
        label_out = f"sfx{i}"

        delay_filters.append(
            f"[{idx}:a]"
            f"adelay={delay_ms}|{delay_ms},"
            f"volume={sfx_volume}"
            f"[{label_out}]"
        )
        mix_labels.append(f"[{label_out}]")

    all_inputs   = "".join(mix_labels)
    n            = len(mix_labels)
    filter_chain = ";".join(delay_filters)
    filter_chain += f";{all_inputs}amix=inputs={n}:duration=first:dropout_transition=0[aout]"

    full_cmd = (
        ["ffmpeg"]
        + cmd
        + [
            "-filter_complex", filter_chain,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]
    )

    result = subprocess.run(full_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[sfx] ffmpeg failed:\n{result.stderr}")
        shutil.copy2(input_path, output_path)
        return

    print(f"[sfx] mixed {len(resolved)} sound effects into video")


def auto_sfx_for_emotion(emotion_type: str, clip_duration: float) -> list[dict]:
    """
    Auto-generates SFX events based on the clip's detected emotion.
    Called when the user does not manually specify SFX.
    """
    events = []

    # Always add a whoosh at the very start — stops the scroll
    events.append({"type": "whoosh", "at_second": 0.0})

    if emotion_type == "excited":
        events.append({"type": "riser",    "at_second": max(0, clip_duration - 3)})

    elif emotion_type == "happy":
        mid = clip_duration / 2
        events.append({"type": "laugh_light", "at_second": mid})

    elif emotion_type in ("sad", "calm"):
        events.append({"type": "love", "at_second": max(0, clip_duration - 2)})

    elif emotion_type == "angry":
        events.append({"type": "impact",   "at_second": 0.5})
        events.append({"type": "bass_drop","at_second": max(0, clip_duration - 2)})

    return events