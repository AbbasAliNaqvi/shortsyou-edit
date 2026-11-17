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

# -14 to -18 dB relative to mastered dialogue: accents, not jump scares.
DEFAULT_SFX_VOLUME = 0.16


def add_sfx(
    input_path: Path,
    output_path: Path,
    events: list[dict],
    sfx_volume: float = DEFAULT_SFX_VOLUME,
) -> None:
    """
    Layers sound effects onto a video at specific timestamps.
    Uses stream copy for video (-c:v copy) for instantaneous 0.2s processing.
    """
    if not events:
        shutil.copy2(input_path, output_path)
        return

    # Resolve which SFX files exist
    resolved = []
    for event in events:
        sfx_type = str(event.get("type", "whoosh")).lower().strip()
        sfx_path = SFX_MAP.get(sfx_type)
        if sfx_path and sfx_path.exists():
            resolved.append({
                "type":      sfx_type,
                "path":      sfx_path,
                "at_second": max(0.0, float(event.get("at_second", 0))),
                "volume":    min(.2, max(.1, float(event.get("volume", sfx_volume)))),
            })
        else:
            print(f"[sfx] skipping {sfx_type} - file not found or unmapped")

    if not resolved:
        shutil.copy2(input_path, output_path)
        return

    # A cue only earns its place at a true beat.  Enforce one every 3 seconds
    # even when an upstream caller submits a noisy list.
    resolved.sort(key=lambda x: x["at_second"])
    spaced = []
    for item in resolved:
        if not spaced or item["at_second"] - spaced[-1]["at_second"] >= 3.0:
            spaced.append(item)
    resolved = spaced

    # Build FFmpeg inputs list
    # Input 0 = main video, Inputs 1..N = SFX files
    cmd = ["-y", "-nostdin", "-i", str(input_path)]

    seen_paths = {}
    sfx_input_indices = []

    for item in resolved:
        path_str = str(item["path"])
        if path_str not in seen_paths:
            # Main video is input 0, so 1st SFX file is input 1, 2nd is 2, etc.
            seen_paths[path_str] = len(seen_paths) + 1
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
            f"volume={item['volume']}"
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

    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"[sfx] ffmpeg failed: {result.stderr}")
        shutil.copy2(input_path, output_path)
        return

    print(f"[sfx] mixed {len(resolved)} sound effects into video at exact timestamps")


def auto_sfx_for_emotion(emotion_type: str, clip_duration: float) -> list[dict]:
    """
    Intelligent dynamic sound design fallback when no explicit LLM events were supplied.
    Distributes high-impact SFX across key narrative milestones instead of a single static effect.
    """
    events = []

    # 1. Intro Hook (0.0s) - Scroll-stopper
    events.append({"type": "whoosh", "at_second": 0.0})

    if clip_duration < 4.0:
        return events

    # 2. Midpoint Beat (35% into clip) - Insight / Stat / Topic shift
    mid_beat = round(clip_duration * 0.35, 2)
    mid_sfx = "ding" if emotion_type in ("happy", "curious", "excited") else "pop"
    events.append({"type": mid_sfx, "at_second": mid_beat})

    if clip_duration < 8.0:
        return events

    # 3. Pre-climax Riser (3.0s before climax)
    if clip_duration >= 14.0:
        riser_time = round(clip_duration * 0.70 - 2.5, 2)
        if riser_time > mid_beat + 2.0:
            events.append({"type": "riser", "at_second": max(0.0, riser_time)})

    # 4. Climax / Punchline (70-75% into clip)
    climax_time = round(clip_duration * 0.75, 2)
    if emotion_type in ("happy", "humorous"):
        climax_sfx = "laugh_light"
    elif emotion_type == "shocked":
        climax_sfx = "shocked"
    elif emotion_type == "serious":
        climax_sfx = "bass_drop"
    else:
        climax_sfx = "impact"

    events.append({"type": climax_sfx, "at_second": climax_time})

    return events
