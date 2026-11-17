import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _speaker_anchor(video_path: Path, width: int, height: int) -> tuple[int, int]:
    """Return a stable face-biased crop anchor, falling back to frame centre.

    Sampling avoids an expensive frame-by-frame detector while preventing a
    centered crop from cutting off a speaker positioned at either edge.
    """
    fallback = (width // 2, height // 2)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    if detector.empty() or eye_detector.empty():
        return fallback
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return fallback
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Ten lightweight samples are enough for a stable anchor and are much
    # faster than scanning every frame.  A strict neighbour threshold is
    # crucial: loose Haar detections can mistake textures for faces.
    sample_count = min(10, max(6, frame_count // 90))
    anchors = []
    try:
        for frame_index in np.linspace(0, max(0, frame_count - 1), sample_count, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=8, minSize=(max(36, width // 18), max(36, height // 18)))
            verified = []
            for x, y, w, h in faces:
                # A face-shaped texture is not enough. Require at least one
                # eye-like feature within it; this prevents foliage, posters,
                # and high-contrast backgrounds from hijacking the crop.
                eyes = eye_detector.detectMultiScale(
                    gray[y:y + h, x:x + w],
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(max(8, w // 7), max(8, h // 7)),
                )
                if len(eyes):
                    verified.append((x, y, w, h))
            if verified:
                x, y, w, h = max(verified, key=lambda face: face[2] * face[3])
                anchors.append((x + w // 2, y + h // 2))
    finally:
        capture.release()
    if len(anchors) < 2:
        return fallback
    # Median rejects a transient listener/cutaway face and keeps the crop calm.
    return int(np.median([a[0] for a in anchors])), int(np.median([a[1] for a in anchors]))


def crop_to_vertical(
    input_path: Path,
    output_path: Path,
    target_w: int = 1080,
    target_h: int = 1920,
):
    width, height = get_dimensions(input_path)

    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read dimensions from {input_path}")

    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target dimensions must be greater than zero")

    target_ratio = target_w / target_h
    source_ratio = width / height

    if source_ratio > target_ratio:
        crop_h = height
        crop_w = int(round(height * target_ratio))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        face_x, _ = _speaker_anchor(input_path, width, height)
        x_offset = face_x - crop_w // 2
        y_offset = 0
    else:
        crop_w = width
        crop_h = int(round(width / target_ratio))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        x_offset = 0
        _, face_y = _speaker_anchor(input_path, width, height)
        # Keep a little headroom over the speaker rather than centering their
        # face exactly in the frame.
        y_offset = face_y - int(crop_h * .35)

    crop_w = max(2, min(crop_w, width))
    crop_h = max(2, min(crop_h, height))
    x_offset = max(0, min(x_offset, width - crop_w))
    y_offset = max(0, min(y_offset, height - crop_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        (
            f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},"
            f"scale={target_w}:{target_h}:flags=lanczos,"
            "format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-profile:v",
        "high",
        "-level",
        "4.2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg crop failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"FFmpeg did not create cropped video: {output_path}"
        )

    if output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Cropped video is empty: {output_path}"
        )

    output_width, output_height = get_dimensions(output_path)

    if output_width != target_w or output_height != target_h:
        raise RuntimeError(
            f"Unexpected cropped dimensions: "
            f"{output_width}x{output_height}, "
            f"expected {target_w}x{target_h}"
        )


def get_dimensions(video_path: Path) -> tuple[int, int]:
    if not video_path.exists():
        return 0, 0

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return 0, 0

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0, 0

    streams = data.get("streams", [])

    if not streams:
        return 0, 0

    stream = streams[0]

    return (
        int(stream.get("width") or 0),
        int(stream.get("height") or 0),
    )
